import base64

from concurrent.futures import TimeoutError
from langchain_ollama import OllamaEmbeddings
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_ollama import ChatOllama
from langsmith import traceable
from pydantic import BaseModel, Field
from threading import local
from typing import List, Tuple, Dict, Any, Optional
from weaviate.classes.data import DataObject

from config.settings import (
    OLLAMA_BASE_URL,
    LLM_CALL_TIMEOUT_SECONDS,
)
from utils.file_utils import extract_content_from_zip, extract_content_from_tar, extract_content_from_rar
from utils.jira_client import JiraIssue, JiraAttachment, JiraClient, JiraRelatedIssue, JiraComment

LOG_FILE_TYPES = ('.log', '.txt', '.out', '.err', '.trace', '.debug', '.zip', '.tar', '.gz', '.tgz', '.tar.gz', '.rar')
IMAGE_FILE_TYPES = ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')

# Ticket processing-specific constants
MAX_LOG_FILES_TO_PROCESS = 20
MAX_LINES_PER_LOG = 50
MAX_LOGS_PER_LLM_CALL = 3
CONTEXT_LINES_BEFORE_ERROR = 10
CONTEXT_LINES_AFTER_ERROR = 20
MAX_WORDS_IN_COMMENTS = 400

AUTOMATION_FOR_JIRA_COMMENT_AUTHOR = "Automation for Jira"


_thread_local = local()


class ImageAnalysisOutput(BaseModel):
    """Structured output for image analysis."""
    error_messages: Optional[str] = Field(default=None, description="Visible error messages in the image")
    summary: str = Field(description="Brief summary of what is visible in the image")


class ErrorInLog(BaseModel):
    """Structured representation of an error found in logs."""
    source_code_filename: Optional[str] = Field(
        description="Python source file (.py) mentioned in the error, e.g. 'main.py'. Leave null if none."
    )
    error_lines: str = Field(
        description="Copy the EXACT line(s) from the log that contain 'ERROR'. Do not paraphrase or summarize."
    )
    context: str = Field(
        description="What went wrong in 1-2 sentences. Example: 'Database connection failed due to timeout.'"
    )


class LogAnalysisOutput(BaseModel):
    """Structured output for log analysis."""
    log_filename: str = Field(description="Name of the log file (.log/.txt suffix)")
    errors: List[ErrorInLog] = Field(min_length=1,
                                     description="List of extracted errors with context. "
                                                 "Must contain at least one error if ERROR "
                                                 "lines exist in the input.")


class FinalIssueSummeryOutput(BaseModel):
    issue_summery: str = Field(description="up to 4 sentences final concise summary of the issue")
    main_issues: List[str] = Field(description="A list of the main issues found in the jira ticket")
    likely_root_causes: List[str] = Field(description="A list of concise root causes if explicitly stated, otherwise 'unknown'")
    comments: str = Field(description="steps taken, key points, and any additional relevant info from comments")


class JiraIssueLLMProcessor:
    """
    Processes Jira issues using LLMs and VLMs to generate summaries and extract information.
    """

    def __init__(self, llm_model_name, vlm_model_name, embedding_model_name = None):
        self.jira_client = JiraClient()
        self.llm_model_name = llm_model_name
        self.vlm_model_name = vlm_model_name
        if embedding_model_name:
            self.embedding_model = OllamaEmbeddings(
                model=embedding_model_name,
                base_url=OLLAMA_BASE_URL
            )

    def process_issue(self, jira_issue: JiraIssue, status_callback: Any = None) -> Tuple[str, List[LogAnalysisOutput]]:
        """
        Process a single Jira issue into LLM-generated super-summary.
        returns a tuple:
        - first element: summary of the issue after processing attachments, comments, and related issues.
        - second element: list of structured log analysis outputs.
        """
        attachments = jira_issue.attachments or []
        comments = jira_issue.comments or []
        related_issues = jira_issue.related_issues or []

        image_summaries, log_summaries = self._process_attachments(attachments, jira_issue.key, status_callback)
        log_summaries_as_text = self._parse_log_analysis_output_to_text(log_summaries)
        image_summaries_as_text = self._parse_image_analysis_output_to_text(image_summaries)
        summarized_attachments = f"{log_summaries_as_text}\n\n{image_summaries_as_text}"
        comments_text = self._process_comments(comments)
        related_issues_text = self._process_related_issues(related_issues)
        # TODO: think about github links in the future

        if status_callback:
            # update status for UI
            status_callback.markdown("🧠 **Generating ticket summary...**")
        issue_summary: FinalIssueSummeryOutput = self._create_final_issue_summary(jira_issue,
                                                                                  comments_text,
                                                                                  summarized_attachments,
                                                                                  related_issues_text)
        issue_summary_as_text = self._parse_final_issue_summary_output_to_text(issue_summary, log_summaries)

        return issue_summary_as_text, log_summaries

    def _get_llm(self) -> ChatOllama:
        # Reuse one LLM client per thread to reduce socket churn
        if not hasattr(_thread_local, "llm") or _thread_local.llm is None:
            _thread_local.llm = ChatOllama(
                model=self.llm_model_name,
                base_url=OLLAMA_BASE_URL,
                temperature=0.1,
            )
        return _thread_local.llm

    def _get_vlm(self) -> ChatOllama:
        # Reuse one VLM client per thread to reduce socket churn
        if not hasattr(_thread_local, "vlm") or _thread_local.vlm is None:
            _thread_local.vlm = ChatOllama(
                model=self.vlm_model_name,
                base_url=OLLAMA_BASE_URL,
                temperature=0.1,
            )
        return _thread_local.vlm

    def _process_attachments(self, attachments: List[JiraAttachment], issue_key: str, status_callback: Any = None) -> (
            Tuple)[List[str], List[LogAnalysisOutput]]:
        """
        Process attachments (images and logs) and return their summaries.
        """
        image_summaries = []
        log_summaries = []

        # Count totals first
        image_attachments = [a for a in attachments if a.filename.lower().endswith(IMAGE_FILE_TYPES)]
        log_attachments = [a for a in attachments if a.filename.lower().endswith(LOG_FILE_TYPES)]

        for idx, attachment in enumerate(image_attachments):
            if status_callback:
                # update status for UI
                status_callback.markdown(f"🖼️ **Analyzing images and screenshots... ({idx + 1}/{len(image_attachments)})**")
            image_summaries.append(self._process_image_attachment(attachment))
        for idx, attachment in enumerate(log_attachments):
            if status_callback:
                # update status for UI
                status_callback.markdown(f"📋 **Analyzing log files... can take some time... ({idx + 1}/{len(log_attachments)})**")
            log_summaries += self._process_log_attachment(attachment, issue_key)

        return image_summaries, log_summaries

    @traceable
    def _process_image_attachment(self, attachment: JiraAttachment) -> str:
        """Process image attachment using llm model to summarize and extract it into text"""
        try:
            image = JiraClient().download_attachment(attachment.id)
            # Convert image bytes to base64 string
            image_b64 = base64.b64encode(image).decode("utf-8")
            image_system_prompt = self._create_image_system_prompt()
            image_user_prompt = self._create_image_user_prompt()

            content_parts = []
            text_part = {"type": "text", "text": image_user_prompt}
            content_parts.append(text_part)
            image_part = {
                "type": "image_url",
                "image_url": f"data:image/jpeg;base64,{image_b64}",
            }
            content_parts.append(image_part)
            vlm = self._get_vlm()
            structured_vlm = vlm.with_structured_output(ImageAnalysisOutput)
            messages = [
                SystemMessage(content=image_system_prompt),
                HumanMessage(content=content_parts)
            ]
            image_summarization = structured_vlm.invoke(messages).summary.strip()

            return image_summarization
        except Exception as e:
            print(f"   ❌ Error processing image attachment {attachment.filename}: {e}")
            return ""

    @staticmethod
    def _create_image_system_prompt() -> str:
        """Create system prompt for image analysis"""
        image_system_prompt = """
            You are an AI model specialized in analyzing screenshots from Jira tickets.
            Rules:
            - Be concise and factual
            - Do not hallucinate
            - If something is unclear or not visible, say "unknown"
            - Do not guess causes without visual evidence
            - Focus only on what is visible in the image


            Be concise and factual. No hallucinations. If unknown, say unknown.
            """
        return image_system_prompt

    @staticmethod
    def _create_image_user_prompt() -> str:
        """Create user prompt for image analysis"""
        image_user_prompt = """
            Analyze the provided screenshot from a Jira ticket.
            Your task is to extract and summarize key information from images, focusing on:
            - visible error messages
            - UI text or warnings
            - any system dialogs
            
            Rules:
            - don't try to understand the root cause, only describe what you see.
            - answer should be under 100 words.
            """
        return image_user_prompt

    def _process_log_attachment(self, attachment: JiraAttachment, issue_key: str) -> (
            Optional)[List[LogAnalysisOutput]]:
        """Process log attachment to extract meaningful text"""
        logs = self._extract_logs_from_file(attachment, issue_key)
        filtered_logs = self._filter_noise_from_logs(logs)
        summarized_logs = self._summarize_filtered_logs(filtered_logs)
        return summarized_logs

    @staticmethod
    def _extract_logs_from_file(attachment: JiraAttachment, issue_key: str) -> List[Tuple[str, str]]:
        """
        Extract log contents from attachment file based on its type.
        returns a list of tuples (log_text, filename)
        """
        suffix = attachment.filename.lower().split('.')[-1]
        log_contents = []

        try:
            file = JiraClient().download_attachment(attachment.id)
        except Exception as e:
            print(f"   ❌ Error downloading log attachment {attachment.filename} for {issue_key}: {e}")
            return []

        if suffix == 'zip':
            try:
                log_contents = extract_content_from_zip(file, MAX_LOG_FILES_TO_PROCESS)
            except Exception as e:
                print(f"   ❌ Error processing log attachment {attachment.filename} for {issue_key}: {e}")
                return []

        if suffix in ('tar', 'gz', 'tgz'):
            try:
                log_contents = extract_content_from_tar(file, suffix, MAX_LOG_FILES_TO_PROCESS)
            except Exception as e:
                print(f"   ❌ Error processing tar attachment {attachment.filename} for {issue_key}: {e}")

        if suffix == 'rar':
            try:
                log_contents = extract_content_from_rar(file, MAX_LOG_FILES_TO_PROCESS)
            except Exception as e:
                print(f"   ❌ Error processing rar attachment {attachment.filename} for {issue_key}: {e}")

        if suffix == 'log' or suffix == 'txt':
            try:
                text = file.decode("utf-8", errors="ignore")
                log_contents.append((text, attachment.filename))
            except Exception as e:
                print(f"   ❌ Error processing log attachment {attachment.filename}: {e}")
                return []

        return log_contents

    @staticmethod
    def _filter_noise_from_logs(log_contents: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
        """
        Extracts only lines containing ERROR (uppercase) and surrounding context lines.
        Deduplicates overlapping segments.
        Caps total output lines.
        Start from the newest log entries (bottom up).
        """
        filtered_logs = []

        for log_text, filename in log_contents:
            lines = log_text.splitlines()
            num_lines = len(lines)

            # Find all error line indexes - only match uppercase ERROR
            # Start from bottom (newest entries)
            error_indexes = []
            for i in range(num_lines - 1, -1, -1):
                if "ERROR" in lines[i]:
                    error_indexes.append(i)

            if not error_indexes:
                continue

            used = set()
            lines_added = 0

            # Process errors from newest to oldest
            for idx in error_indexes:
                start = max(0, idx - CONTEXT_LINES_BEFORE_ERROR)
                end = min(num_lines, idx + CONTEXT_LINES_AFTER_ERROR + 1)

                for i in range(start, end):
                    if i not in used:
                        used.add(i)
                        lines_added += 1

                # Stop collecting if we've hit the limit
                if lines_added >= MAX_LINES_PER_LOG:
                    break

            # Sort indices to maintain chronological order in output
            extracted = ["Extracted error context:\n"]
            for i in sorted(used):
                line = lines[i]

                if len(line) > 2000:
                    continue

                extracted.append(line)

            if len(extracted) > MAX_LINES_PER_LOG:
                extracted = extracted[:MAX_LINES_PER_LOG]
                extracted.append(f"... truncated after {MAX_LINES_PER_LOG} lines ...")

            filtered_logs.append(("\n".join(extracted), filename))

        return filtered_logs

    def _summarize_filtered_logs(self, filtered_logs: List[Tuple[str, str]]) -> (
            Optional[List[LogAnalysisOutput]]):
        """Summarize filtered logs using LLM,"""
        if not filtered_logs:
            return []

        all_summaries = []
        for log in filtered_logs:
            summary = self._summarize_log(log)
            summary = self._filter_out_unwanted_logs(summary)
            if summary:
                if summary.errors:
                    all_summaries.append(summary)

        return all_summaries

    @traceable
    def _summarize_log(self, log: Tuple[str, str]) -> Optional[LogAnalysisOutput]:
        """Summarize a log file"""
        system_prompt = self._create_log_summery_system_prompt()
        log_summery_prompt = self._create_log_summery_prompt(log)

        llm = self._get_llm()
        structured_llm = llm.with_structured_output(LogAnalysisOutput)
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=log_summery_prompt)
        ]
        try:
            response = structured_llm.invoke(messages)
        except TimeoutError:
            print(f"   ❌ LLM call timed out after {LLM_CALL_TIMEOUT_SECONDS} seconds for log {log[1]}")
            return None

        if isinstance(response, dict):
            response = LogAnalysisOutput(**response)

        response.log_filename = log[1]
        return response

    @staticmethod
    def _parse_log_analysis_output_to_text(logs_analysis: List[LogAnalysisOutput]) -> str:
        """Convert structured log analysis output to text format"""
        if not logs_analysis:
            return "No errors found in logs."

        result_lines = ["**Errors:**"]
        for log_analysis in logs_analysis:
            for error in log_analysis.errors:
                result_lines.append(f"- Log File: `{log_analysis.log_filename}`")
                if error.source_code_filename:
                    result_lines.append(f"  File in code: `{error.source_code_filename}`")
                result_lines.append(f"  Error: \"{error.error_lines}\"")
                result_lines.append(f"  Context: {error.context}")
                result_lines.append("")  # Add an empty line for better readability

        return "\n".join(result_lines).strip()

    @staticmethod
    def _filter_out_unwanted_logs(summary: LogAnalysisOutput) -> Optional[LogAnalysisOutput]:
        """Filter out summaries that don't contain real errors"""
        if not summary:
            return None

        filtered_errors = [
            error_entry for error_entry in summary.errors
            if "ERROR" in error_entry.error_lines
        ]
        summary.errors = filtered_errors
        return summary

    @traceable
    def _aggregate_log_summaries(self, summaries: List[str]) -> str:
        """Aggregate multiple log batch summaries into one"""
        system_prompt = """You merge multiple error extraction results into a single consolidated report.

    RULES:
    - Combine all errors from all inputs
    - Remove duplicates (same error in same file)
    - Keep the exact format: File, Error, Context
    - List patterns only once at the end
    - Do NOT add commentary"""

        user_prompt = f"""Merge these error reports into one consolidated report.
    
    ---
    
    Use this format:
    
    **Errors:**
    - File: `[log filename (files with .log/.txt suffix)]`
      File in code: `[source code filename (files with .py suffix)]` if exists. if not, skip this line.
      Error: "[error lines]"
      Context: [description]

    ---
    REPORTS TO MERGE:
    {chr(10).join(summaries)}
    """

        llm = self._get_llm()
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        response = llm.invoke(messages).content
        return response.strip()

    @staticmethod
    def _create_log_summery_system_prompt() -> str:
        """Create system prompt for log summarization"""
        log_summery_system_prompt = """You are a log parser that extracts ERROR lines from application logs.

        STRICT RULES:
        1. Find every line containing "ERROR" (uppercase)
        2. For error_lines: Copy the ENTIRE log line verbatim, exactly as written
        3. For context: Write 1-2 sentences explaining what the error means
        4. For source_code_filename: Only include .py files mentioned in the error
        5. Deduplicate errors with minor differences (e.g., timestamps, thread IDs) if they are otherwise identical.
           - Merge such errors into one entry, and include the most recent timestamp.
           - Ensure the context remains accurate and relevant.

        
        EXAMPLE INPUT:
        2024-01-15 10:23:45 ERROR controller.py:142 - Failed to connect to database: timeout
        2024-01-15 10:24:00 ERROR controller.py:142 - Failed to connect to database: timeout
        
        EXAMPLE OUTPUT:
        - error_lines: "2024-01-15 10:24:00 ERROR controller.py:142 - Failed to connect to database: timeout"
        - source_code_filename: "controller.py"
        - context: "Database connection failed due to a timeout error."
        
        remove any duplicate errors found in the same log file.
        DO NOT summarize or shorten error_lines. Copy exactly."""

        return log_summery_system_prompt

    @staticmethod
    def _create_log_summery_prompt(filtered_logs: Tuple[str, str]) -> str:
        """Create user prompt for log summarization"""

        return f"""Parse these log snippets and extract ALL error lines.
        
        Remember:
        - error_lines = exact copy of the ERROR line from above
        - context = brief explanation of what went wrong
        - source_code_filename = any .py file mentioned (or null)
                
        LOG TO ANALYZE:
        {filtered_logs[0]}
        """

    @staticmethod
    def _parse_image_analysis_output_to_text(image_summaries: List[str]) -> str:
        """Convert structured image analysis output to text format"""
        if not image_summaries:
            return "No images to analyze."

        result_lines = ["**Images:**"]
        for idx, image_summary in enumerate(image_summaries):
            result_lines.append(f"image {idx + 1} summary:")
            result_lines.append(f"{image_summary}")
            result_lines.append("")  # Add an empty line for better readability

        return "\n".join(result_lines).strip()

    @traceable
    def _aggregate_attachment_summaries(self, image_summaries: List[str], log_summaries: List[str]) -> str:
        """Aggregate multiple attachment summaries into one final summary"""
        aggregate_attachments_system_prompt = self._create_aggregate_attachments_system_propmt()
        aggregate_attachments_prompt = self._create_aggregate_attachments_propmt(image_summaries, log_summaries)
        llm = self._get_llm()
        messages = [
            SystemMessage(content=aggregate_attachments_system_prompt),
            HumanMessage(content=aggregate_attachments_prompt)
        ]

        return llm.invoke(messages).content.strip()

    @staticmethod
    def _create_aggregate_attachments_system_propmt() -> str:
        """Create system prompt for aggregating attachment summaries"""
        aggregate_attachments_system_propmt = """
        You summarize attachment analyses from Jira tickets.

        Rules:
        - Be concise and factual
        - Do not hallucinate or invent details
        - Only use information explicitly present in the input summaries
        - If information is missing or unclear, say "unknown"
        - Do not repeat the same point multiple times
        - Merge similar findings into a single clear statement
        - Do not speculate about root causes unless they were explicitly stated in the input
        - Keep the result short and readable
        """
        return aggregate_attachments_system_propmt

    @staticmethod
    def _create_aggregate_attachments_propmt(image_summaries: List[str], log_summaries: List[str]) -> str:
        """Create user prompt for aggregating attachment summaries"""
        aggregate_attachments_propmt = f"""
        Summarize the following attachment summaries into a single consolidated summary.

        The input consists of summaries produced from:
        - log files
        - screenshots
        
        Your task:
        - Combine overlapping findings
        - Highlight key errors and warnings
        - Mention all errors found in the logs exactly as they are, with their context - the filename of the log 
        (.log, .txt files), the filename of where the error was found in the code (.py files), and the understanding of 
        the error itself
        - Mention any explicitly stated root causes
        - Mention affected components/services/log files
        - Keep the description of the images
        
        ---
        
        YOU MUST USE THIS EXACT FORMAT:

        **Errors:**
        - File: `[filename from bracket prefix - log file name and source code file name]`
          Error: "[copy the full ERROR line exactly]"
          Context: [one sentence]
          
        **Images:**
        - [summary of 1-2 sentences per image summary]
        - key points from screenshots
        
        ---
        
        Image summaries:
        {image_summaries}
        
        Log summaries:
        {log_summaries}
        """
        return aggregate_attachments_propmt

    @staticmethod
    def _process_comments(comments: List[JiraComment]) -> str:
        """Concatenate up to MAX_WORDS_IN_COMMENTS words of the most recent comments into a single string."""
        # Sort comments by creation date descending (most recent first)
        if not comments:
            return "No comments in the ticket."

        sorted_comments = sorted(comments, key=lambda c: getattr(c, "created", ""), reverse=True)
        result = []
        word_count = 0
        for comment in sorted_comments:
            if comment.author_display_name == AUTOMATION_FOR_JIRA_COMMENT_AUTHOR:
                continue  # skip comments from Automation for Jira
            text = f"{comment.author_display_name}: {comment.body}"
            words = text.split()
            if word_count + len(words) > MAX_WORDS_IN_COMMENTS:
                # Only add up to the word limit
                remaining = MAX_WORDS_IN_COMMENTS - word_count
                result.append(" ".join(words[:remaining]))
                break
            result.append(text)
            word_count += len(words)
        return "\n".join(result)

    @staticmethod
    def _process_related_issues(related_issues: List[JiraRelatedIssue]) -> str:
        """Concatenate related issues into a single string"""
        if not related_issues:
            return "No related issues for the ticket."
        related_issues_text = "\n".join(
            [f"{rel_issue.direction}:{rel_issue.key}({rel_issue.relation_type})" for rel_issue in related_issues]
        )

        return related_issues_text

    def create_data_object(self, base_props: Dict, content: str) -> DataObject:
        """Create a Weaviate data object"""

        properties = base_props.copy()

        # Generate embedding
        vector = self.embedding_model.embed_query(content)

        return DataObject(
            properties=properties,
            vector=vector
        )

    @traceable
    def _create_final_issue_summary(
            self,
            jira_issue: JiraIssue,
            comments_text: str,
            summarized_attachments: str,
            related_issues_text: str
    ) -> FinalIssueSummeryOutput:
        """Create the final issue summary combining all components"""
        final_issue_summery_system_prompt = self._create_final_issue_summery_system_prompt()
        final_issue_summery_prompt = self._create_final_issue_summery_prompt(
            jira_issue, comments_text, summarized_attachments, related_issues_text
        )
        llm = self._get_llm()
        structured_llm = llm.with_structured_output(FinalIssueSummeryOutput)
        messages = [
            SystemMessage(content=final_issue_summery_system_prompt),
            HumanMessage(content=final_issue_summery_prompt)
        ]

        result = structured_llm.invoke(messages)
        if isinstance(result, dict):
            result = FinalIssueSummeryOutput(**result)

        return result

    @staticmethod
    def _create_final_issue_summery_system_prompt() -> str:
        """Create system prompt for final issue summarization"""
        final_issue_summery_system_prompt = """
        You generate final summaries for Jira tickets.
        
        Rules:
        - Be concise, factual, and neutral
        - Do not hallucinate or invent details
        - Use only the information provided
        - If something is unclear or missing, say "unknown"
        - Do not repeat the same information
        - Merge related findings into a single clear statement
        - Do not speculate about root causes unless explicitly supported by the input
        - Write in professional technical language suitable for a Jira issue summary
        """
        return final_issue_summery_system_prompt

    @staticmethod
    def _create_final_issue_summery_prompt(
            jira_issue: JiraIssue,
            comments_text: str,
            summarized_attachments: str,
            related_issues_text: str
    ) -> str:
        """Create user prompt for final issue summarization"""
        final_issue_summery_prompt = f"""
        Create a final consolidated summary for the Jira ticket using the information below.

        Your goal:
        - Provide a clear overview of the issue
        - Highlight confirmed errors or symptoms
        Mention all errors found in the logs exactly as they are, with their context - the filename of the log 
        (.log, .txt files), the filename of where the error was found in the code (.py files), and the understanding of 
        the error itself
        - Mention any explicitly stated root causes
        - Reference related issues only if they add useful context
        
        Inputs:

        Ticket information:
        - Issue Summary: {jira_issue.summary}
        - Issue Description: {jira_issue.description or ''}
        - Issue Key: {jira_issue.key}
        - Issue Labels: {', '.join(jira_issue.labels) if jira_issue.labels else 'None'}
        - Issue Type: {jira_issue.issue_type or 'Unknown'}
        - Issue Priority: {jira_issue.priority or 'Unknown'}
        - Issue Components: {', '.join(jira_issue.components) if jira_issue.components else 'None'}
        - Issue Status: {jira_issue.status or 'Unknown'}
        - Issue Creation Date: {jira_issue.created or 'Unknown'}
        
        
        Issue Comments (chronological, combined): 
        {comments_text}
        
        
        Issue Attachments Summary: 
        {summarized_attachments}
        
        
        Related Issues: 
        {related_issues_text}
        """
        return final_issue_summery_prompt

    @staticmethod
    def _parse_final_issue_summary_output_to_text(issue_summary: FinalIssueSummeryOutput,
                                                  log_summaries: List[LogAnalysisOutput]) -> str:
        """Convert structured final issue summary output to text format"""
        result_lines = [f"**Final Issue Summary:** {issue_summary.issue_summery}\n", "**Main Issues Identified:**"]

        for main_issue in issue_summary.main_issues:
            result_lines.append(f"- {main_issue}")

        result_lines.append("\n**Likely Root Causes:**")
        for root_cause in issue_summary.likely_root_causes:
            result_lines.append(f"- {root_cause}")

        result_lines.append("\n**Errors Found In Logs:**")
        if log_summaries:
            for log_analysis in log_summaries:
                for error in log_analysis.errors:
                    result_lines.append(f"- Log File: `{log_analysis.log_filename}`")
                    if error.source_code_filename:
                        result_lines.append(f"  File in code: `{error.source_code_filename}`")
                    result_lines.append(f"  Error: \"{error.error_lines}\"")
                    result_lines.append(f"  Context: {error.context}")
                    result_lines.append("")  # Add an empty line for better readability
        else:
            result_lines.append("No errors found in logs.")

        result_lines.append(f"\n**Comments Summary:**\n{issue_summary.comments}")

        return "\n".join(result_lines).strip()

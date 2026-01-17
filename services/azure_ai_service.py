"""Azure AI service integration."""
from pathlib import Path
from typing import Dict, Any, Optional, Union
import pandas as pd
import json
import os
import re
import requests
from dotenv import load_dotenv
from openai import AzureOpenAI
from config import (
    DEBUG as APP_DEBUG,
    AZURE_OPENAI_CUSTOM_PROMPT,
    BASE_DIR,
    JSON_GENERATION_METHOD,
    AI_SERVICE,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    OLLAMA_TIMEOUT,
    AZURE_OPENAI_TEMPERATURE,
    OLLAMA_TEMPERATURE
)
from config import AZURE_OPENAI_SYSTEM_PROMPT, AZURE_OPENAI_USER_PROMPT
from config import (
    AZURE_OPENAI_INCLUDE_IMAGES,
    AZURE_OPENAI_INCLUDE_RAW_MINERU_JSON,
    AZURE_OPENAI_RAW_JSON_MAX_CHARS,
)


load_dotenv()


class AzureAIService:
    """AI service integration (supports Azure OpenAI and Ollama)."""
    
    def __init__(self):
        self.ai_service = AI_SERVICE
        self.api_key = os.getenv("AZURE_OPENAI_API_KEY")
        self.api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2023-12-01-preview")
        self.azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        self.deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4")
        self.ollama_base_url = OLLAMA_BASE_URL
        self.ollama_model = OLLAMA_MODEL
        self.ollama_timeout = OLLAMA_TIMEOUT
        self.azure_openai_temperature = AZURE_OPENAI_TEMPERATURE
        self.ollama_temperature = OLLAMA_TEMPERATURE
        self.client = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize the AI client (Azure OpenAI or Ollama)."""
        if self.ai_service == "ollama":
            if APP_DEBUG:
                print("[AI_SERVICE] Using Ollama service")
                print(f"[AI_SERVICE]   Model: {self.ollama_model}")
                print(f"[AI_SERVICE]   Endpoint: {self.ollama_base_url}")
            # Ollama does not need a client instance; we use HTTP requests directly.
            self.client = "ollama"  # marker for Ollama
        else:
            # Use Azure OpenAI.
            if self.api_key and self.azure_endpoint:
                try:
                    self.client = AzureOpenAI(
                        api_key=self.api_key,
                        api_version=self.api_version,
                        azure_endpoint=self.azure_endpoint
                    )
                    if APP_DEBUG:
                        print("[AI_SERVICE] Using Azure OpenAI service")
                        print(f"[AI_SERVICE]   Deployment: {self.deployment_name}")
                        print(f"[AI_SERVICE]   Endpoint: {self.azure_endpoint}")
                except Exception as e:
                    print(f"Warning: Azure OpenAI client initialization failed: {e}")
                    self.client = None
            else:
                if APP_DEBUG:
                    print("Warning: Azure OpenAI credentials are not set; using simple extraction")
                self.client = None
    
    def _call_ollama(self, system_prompt: str, user_prompt: str, temperature: float = None) -> str:
        """
        Call the Ollama API.
        
        Args:
            system_prompt: System prompt
            user_prompt: User prompt
            temperature: Temperature (if None, uses self.ollama_temperature)
        
        Returns:
            AI response text
        """
        if temperature is None:
            temperature = self.ollama_temperature
        try:
            if APP_DEBUG:
                print(f"[OLLAMA] Calling Ollama API: {self.ollama_base_url}")
                print(f"[OLLAMA]   Model: {self.ollama_model}")
            
            # Ollama API endpoint.
            url = f"{self.ollama_base_url}/api/chat"
            
            # Build request payload.
            payload = {
                "model": self.ollama_model,
                "messages": [
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": user_prompt
                    }
                ],
                "stream": False,
                "options": {
                    "temperature": temperature
                }
            }
            
            # Send request.
            if APP_DEBUG:
                print(f"[OLLAMA]   Timeout: {self.ollama_timeout} seconds")
            response = requests.post(url, json=payload, timeout=self.ollama_timeout)
            response.raise_for_status()
            
            result = response.json()
            
            # Extract response content.
            if "message" in result and "content" in result["message"]:
                content = result["message"]["content"]
                if APP_DEBUG:
                    print(f"[OLLAMA] ✓ Response received (length: {len(content)} chars)")
                return content
            else:
                if APP_DEBUG:
                    print(f"[OLLAMA] ⚠ Unexpected response format: {result}")
                return ""
                
        except requests.exceptions.RequestException as e:
            if APP_DEBUG:
                print(f"[OLLAMA] ✗ Request failed: {e}")
            raise Exception(f"Ollama API call failed: {str(e)}")
        except Exception as e:
            if APP_DEBUG:
                print(f"[OLLAMA] ✗ Processing failed: {e}")
            raise Exception(f"Ollama processing failed: {str(e)}")
    
    async def generate_csv_from_json(
        self, 
        mineru_json: Union[Dict[str, Any], Dict[str, Optional[Path]]], 
        output_path: Path,
        progress_callback=None,
        output_format: str = "json",
        system_prompt_override: Optional[str] = None,
        user_prompt_override: Optional[str] = None,
        markdown_content: Optional[str] = None,
        image_data_urls: Optional[list[str]] = None,
    ) -> Path:
        """
        Generate structured output (JSON) from MinerU JSON or PDFs.

        The generation method depends on configuration:
        1. If input is a PDF-path dict (contains layout_pdf or span_pdf), extract content from PDFs.
        2. If JSON_GENERATION_METHOD=mineru_csv: simple extraction (from JSON only; no AI calls).
        3. If JSON_GENERATION_METHOD=azure_openai_csv: AI-assisted extraction (requires AI service).
        4. If there is no AI client: automatically fall back to simple extraction.
        """
        # Detect whether input is a PDF-path dictionary.
        if isinstance(mineru_json, dict) and ("layout_pdf" in mineru_json or "span_pdf" in mineru_json):
            if APP_DEBUG:
                print("[JSON_GENERATION] Detected PDF input; extracting content from PDFs")
            # Extract content from PDFs.
            mineru_json = await self._extract_content_from_pdfs(mineru_json)
        
        # Ensure output_path stays under the configured output folder (avoid writing elsewhere).
        # - JSON outputs -> JSON_OUTPUT_DIR
        # - CSV outputs (suffix .csv) -> CSV_OUTPUT_DIR
        try:
            from config import JSON_OUTPUT_DIR, CSV_OUTPUT_DIR
            output_path = Path(output_path)
            safe_root = CSV_OUTPUT_DIR if (output_path.suffix or "").lower() == ".csv" else JSON_OUTPUT_DIR
            # If output_path is outside safe_root, relocate to safe_root/<name>.
            try:
                if not str(output_path.resolve()).startswith(str(Path(safe_root).resolve())):
                    output_path = Path(safe_root) / output_path.name
            except Exception:
                output_path = Path(safe_root) / output_path.name
        except Exception:
            # If output dirs cannot be imported, still coerce to Path.
            output_path = Path(output_path)
        # Select generation/extraction method.
        use_mineru_csv = JSON_GENERATION_METHOD == "mineru_csv"
        use_azure_openai_csv = JSON_GENERATION_METHOD == "azure_openai_csv"
        use_program_csv = JSON_GENERATION_METHOD == "program_csv"
        
        # Backward compatibility for legacy names.
        if JSON_GENERATION_METHOD == "simple":
            use_mineru_csv = True
            if APP_DEBUG:
                print("[JSON_GENERATION] ⚠ Detected legacy name 'simple'; consider updating to 'mineru_csv'")
        elif JSON_GENERATION_METHOD == "azure_openai":
            use_azure_openai_csv = True
            if APP_DEBUG:
                print("[JSON_GENERATION] ⚠ Detected legacy name 'azure_openai'; consider updating to 'azure_openai_csv'")
        
        # If program_csv is selected, use programmatic flattening.
        if use_program_csv:
            if APP_DEBUG:
                print("[JSON_GENERATION] Using program_csv extraction (programmatic flattening)")
            if progress_callback:
                await progress_callback(10, "Parsing JSON…", "program_extract")
            result = await self._program_csv_extraction(mineru_json, output_path)
            if progress_callback:
                await progress_callback(90, "Generating output…", "program_complete")
            return result
        
        # If mineru_csv is selected, use simple extraction.
        if use_mineru_csv:
            if APP_DEBUG:
                print("[JSON_GENERATION] Using mineru_csv extraction (direct from MinerU JSON)")
                print("[JSON_GENERATION]   Output is generated directly from MinerU JSON (no AI processing)")
            if progress_callback:
                await progress_callback(10, "Parsing content…", "simple_extract")
            result = await self._simple_extraction(mineru_json, output_path)
            if progress_callback:
                await progress_callback(90, "Generating output…", "simple_complete")
            return result
        
        # Default to azure_openai_csv (or fall back to mineru_csv if no client).
        if not use_azure_openai_csv and JSON_GENERATION_METHOD not in ["mineru_csv", "program_csv"]:
            if APP_DEBUG:
                print(f"[JSON_GENERATION] ⚠ Unknown JSON_GENERATION_METHOD: {JSON_GENERATION_METHOD}; using azure_openai_csv")
        
        # If there is no AI client, use simple extraction.
        if not self.client:
            if APP_DEBUG:
                service_name = "Azure OpenAI" if self.ai_service == "azure_openai" else "Ollama"
                print(f"[JSON_GENERATION] ⚠ {service_name} client not initialized; using simple extraction")
                if self.ai_service == "azure_openai":
                    print("[JSON_GENERATION]   Reason: Azure OpenAI credentials are missing or initialization failed")
                else:
                    print("[JSON_GENERATION]   Reason: Ollama service is not running or connection failed")
                print("[JSON_GENERATION]   Output will not include AI-processed structured data")
            return await self._simple_extraction(mineru_json, output_path)
        
        try:
            if APP_DEBUG:
                    if self.client == "ollama":
                        print("[JSON_GENERATION] ✓ Generating output via Ollama")
                        print(f"[JSON_GENERATION]   Model: {self.ollama_model}")
                        print(f"[JSON_GENERATION]   Endpoint: {self.ollama_base_url}")
                        print(f"[JSON_GENERATION]   Temperature: {self.ollama_temperature} (0=deterministic output for consistent formatting)")
                    else:
                        print("[JSON_GENERATION] ✓ Generating output via Azure OpenAI")
                        print(f"[JSON_GENERATION]   Deployment: {self.deployment_name}")
                        print(f"[JSON_GENERATION]   API version: {self.api_version}")
                        print(f"[JSON_GENERATION]   Endpoint: {self.azure_endpoint}")
                        print(f"[JSON_GENERATION]   Temperature: {self.azure_openai_temperature} (0=deterministic output for consistent formatting)")
            
            # Send progress update.
            if progress_callback:
                await progress_callback(20, "Preparing prompts…", "prepare_prompt")
            
            # Load custom prompts (supports system_prompt and user_prompt).
            custom_system_prompt, custom_user_prompt = self._load_custom_prompts(system_override=system_prompt_override, user_override=user_prompt_override)

            # Prepare the system prompt (priority: frontend override -> .env -> file -> default).
            if custom_system_prompt:
                system_prompt = custom_system_prompt
                if APP_DEBUG:
                    print(f"[JSON_GENERATION]   Using custom system_prompt (length: {len(system_prompt)} chars)")
            else:
                # Output is fixed to JSON: the default prompt must force JSON to avoid markdown/tables.
                system_prompt = (
                    "You are a data-extraction assistant. Return ONLY valid JSON (no markdown, no code fences).\n"
                    "If a value is unknown, use null. Use the exact keys requested by the user prompt."
                )
                if APP_DEBUG:
                    print("[JSON_GENERATION]   Using default system_prompt")

            # Prepare the user prompt (priority: frontend override -> .env -> file -> auto-generated).
            if custom_user_prompt is not None:
                # If user_prompt is provided (even an empty string), use it.
                if custom_user_prompt.strip():
                    # Non-empty user_prompt; replace placeholders (if any).
                    prompt = self._create_extraction_prompt_from_template(mineru_json, custom_user_prompt) if not markdown_content else custom_user_prompt
                    if APP_DEBUG:
                        print(f"[JSON_GENERATION]   Using custom user_prompt (length: {len(prompt)} chars)")
                else:
                    # Empty user_prompt: use content only as the user prompt.
                    if markdown_content:
                        prompt = markdown_content
                    else:
                        content_dict = self._extract_content_from_json(mineru_json)
                        prompt = content_dict.get('content', '')
                    if APP_DEBUG:
                        print(f"[JSON_GENERATION]   user_prompt is empty; using content only (length: {len(prompt)} chars)")
            else:
                # No user_prompt provided: use default prompt builder or markdown content.
                if markdown_content:
                    prompt = markdown_content
                else:
                    prompt = self._create_extraction_prompt(mineru_json)
                if APP_DEBUG:
                    print("[JSON_GENERATION]   Using default user_prompt")

            # Optionally append raw MinerU JSON (can be large).
            try:
                include_raw = bool(AZURE_OPENAI_INCLUDE_RAW_MINERU_JSON)
                if include_raw and mineru_json is not None and not markdown_content:
                    raw = json.dumps(mineru_json, ensure_ascii=False)
                    max_chars = int(AZURE_OPENAI_RAW_JSON_MAX_CHARS or 0)
                    if max_chars > 0 and len(raw) > max_chars:
                        raw = raw[:max_chars] + "\n... [TRUNCATED]\n"
                        if APP_DEBUG:
                            print(f"[JSON_GENERATION] ⚠ Raw MinerU JSON truncated to {max_chars} chars")
                    prompt = (
                        prompt
                        + "\n\n[RAW_MINERU_JSON]\n"
                        + raw
                    )
            except Exception as e:
                if APP_DEBUG:
                    print(f"[JSON_GENERATION] ⚠ Failed to append raw MinerU JSON: {e}")
            
            if APP_DEBUG:
                prompt_length = len(prompt)
                system_prompt_length = len(system_prompt)
                print(f"[JSON_GENERATION]   User prompt length: {prompt_length} chars")
                print(f"[JSON_GENERATION]   System prompt length: {system_prompt_length} chars")
                print(f"[JSON_GENERATION]   Total prompt length: {prompt_length + system_prompt_length} chars")
                print()
                print("=" * 80)
                if self.client == "ollama":
                    print("[JSON_GENERATION] Full prompt sent to Ollama:")
                else:
                    print("[JSON_GENERATION] Full prompt sent to Azure OpenAI:")
                print("=" * 80)
                print()
                print("[SYSTEM PROMPT]:")
                print("-" * 80)
                print(system_prompt)
                print()
                print("[USER PROMPT]:")
                print("-" * 80)
                # If the prompt is long, print head/tail and omit the middle.
                if len(prompt) > 10000:
                    print("[First 5000 chars]:")
                    print(prompt[:5000])
                    print()
                    print(f"... [omitted middle {len(prompt) - 10000} chars] ...")
                    print()
                    print("[Last 5000 chars]:")
                    print(prompt[-5000:])
                    print()
                    print(f"[Full prompt length: {len(prompt)} chars]")
                else:
                    print(prompt)
                print()
                print("=" * 80)
                if self.client == "ollama":
                    print("[JSON_GENERATION]   Calling Ollama API...")
                else:
                    print("[JSON_GENERATION]   Calling Azure OpenAI API...")
                print()
            
            # Call the AI service (Azure OpenAI or Ollama).
            if self.client == "ollama":
                # Use Ollama.
                if progress_callback:
                    await progress_callback(40, "AI generating…", "call_ollama")
                response_text = self._call_ollama(system_prompt, prompt, self.ollama_temperature)
                if progress_callback:
                    await progress_callback(70, "Parsing AI response…", "parse_ollama")
                # Create a mock response object to maintain compatibility.
                class MockResponse:
                    def __init__(self, content, model_name):
                        self.choices = [type('obj', (object,), {'message': type('obj', (object,), {'content': content})()})()]
                        self.usage = None
                        self.id = "ollama-response"
                        self.model = model_name
                response = MockResponse(response_text, self.ollama_model)
            else:
                # Use Azure OpenAI.
                if progress_callback:
                    await progress_callback(40, "AI generating…", "call_azure_openai")

                # If we have images and multimodal is enabled, send user content as [text + images].
                use_images = bool(AZURE_OPENAI_INCLUDE_IMAGES) and bool(image_data_urls)
                user_content: Any
                if use_images:
                    parts = [{"type": "text", "text": prompt}]
                    for url in image_data_urls or []:
                        if not url:
                            continue
                        parts.append({"type": "image_url", "image_url": {"url": url}})
                    user_content = parts
                    if APP_DEBUG:
                        print(f"[JSON_GENERATION]   Including {len(parts) - 1} image(s) in Azure OpenAI request")
                else:
                    user_content = prompt

                request_kwargs = {
                    "model": self.deployment_name,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_content},
                    ],
                    "temperature": self.azure_openai_temperature,
                }

                # When expecting JSON output, try to hard-enforce JSON response format.
                # Some Azure OpenAI API versions/models may not support this; we retry without it.
                if (output_format or "json").lower() == "json":
                    request_kwargs["response_format"] = {"type": "json_object"}

                try:
                    response = self.client.chat.completions.create(**request_kwargs)
                except Exception as e:
                    # Backward compatible retry if response_format is unsupported.
                    if "response_format" in request_kwargs and ("response_format" in str(e) or "json_object" in str(e)):
                        if APP_DEBUG:
                            print("[JSON_OUTPUT] ⚠ response_format is not supported by this API/model; retrying without it")
                        request_kwargs.pop("response_format", None)
                        response = self.client.chat.completions.create(**request_kwargs)
                    else:
                        raise
                if progress_callback:
                    await progress_callback(70, "Parsing AI response…", "parse_azure_openai")
            
            # Show API response details (DEBUG mode).
            if APP_DEBUG:
                if self.client == "ollama":
                    print("[JSON_GENERATION]   Ollama API response succeeded")
                    print(f"[JSON_GENERATION]   Model: {self.ollama_model}")
                else:
                    # Get token usage (if available).
                    usage = getattr(response, 'usage', None)
                    if usage:
                        prompt_tokens = getattr(usage, 'prompt_tokens', 'N/A')
                        completion_tokens = getattr(usage, 'completion_tokens', 'N/A')
                        total_tokens = getattr(usage, 'total_tokens', 'N/A')
                        print("[JSON_GENERATION]   Azure OpenAI API response succeeded")
                        print(f"[JSON_GENERATION]   Token usage: prompt={prompt_tokens}, completion={completion_tokens}, total={total_tokens}")
                    else:
                        print("[JSON_GENERATION]   Azure OpenAI API response succeeded (token usage unavailable)")
                    
                    # Print response ID (proves it came from Azure OpenAI).
                    response_id = getattr(response, 'id', None)
                    if response_id:
                        print(f"[JSON_GENERATION]   Response ID: {response_id}")
                    
                    # Print model info.
                    model = getattr(response, 'model', None)
                    if model:
                        print(f"[JSON_GENERATION]   Model used: {model}")
            
            # Parse response.
            raw_response = response.choices[0].message.content
            
            if APP_DEBUG:
                response_length = len(raw_response)
                print(f"[JSON_GENERATION]   Received response length: {response_length} chars")
                print()
                print("=" * 80)
                print("[JSON_GENERATION] AI raw response content:")
                print("=" * 80)
                print(raw_response[:2000])  # show first 2000 chars
                if len(raw_response) > 2000:
                    print(f"... ({len(raw_response) - 2000} more chars)")
                print("=" * 80)
                print()
            
            # Try extracting JSON content from the response (prefer code blocks).
            if progress_callback:
                await progress_callback(75, "Processing output format…", "parse_response")

            # Prefer extracting a JSON code block, e.g. ```json {...} ```.
            extracted_json_block = self._extract_codeblock_from_response(raw_response, lang_hint='json')
            parsed_json = None
            if extracted_json_block:
                try:
                    parsed_json = json.loads(extracted_json_block)
                    if APP_DEBUG:
                        print(f"[JSON_OUTPUT] ✓ Parsed JSON from code block (length: {len(extracted_json_block)} chars)")
                except Exception:
                    if APP_DEBUG:
                        print("[JSON_OUTPUT] ⚠ Code block looks like JSON but failed to parse; trying other strategies")

            # If no code block, try parsing the whole response as JSON.
            if parsed_json is None:
                try:
                    parsed_json = json.loads(raw_response)
                    if APP_DEBUG:
                        print(f"[JSON_OUTPUT] ✓ Parsed JSON directly from AI response (length: {len(raw_response)} chars)")
                except Exception:
                    if APP_DEBUG:
                        print("[JSON_OUTPUT] ⚠ Failed to parse AI response as JSON; trying table/free-text parsing")

            # If JSON is still not available:
            # - When output_format=json, do not attempt CSV/table parsing (it can produce misleading CSV tokenizing errors).
            # - Save a clear fallback JSON to make debugging prompts/model output easier.
            if parsed_json is None:
                if (output_format or "json").lower() == "json":
                    parsed_json = {
                        "_warning": "Model did not return valid JSON; saved raw response instead.",
                        "raw_response": raw_response,
                    }
                else:
                    if progress_callback:
                        await progress_callback(80, "Parsing AI response into structured data…", "parse_dataframe")
                    df = self._parse_ai_response_to_dataframe(raw_response, mineru_json, output_format=output_format)
                    parsed_json = df.to_dict(orient='records') if not df.empty else []

            # Write final JSON to output_path (caller is expected to use a .json suffix).
            try:
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(parsed_json, f, ensure_ascii=False, indent=2)
                if APP_DEBUG:
                    print(f"[JSON_OUTPUT] ✓ JSON file generated: {output_path}")
            except Exception as e:
                if APP_DEBUG:
                    print(f"[JSON_OUTPUT] ✗ Failed to write JSON file: {e}")
                raise

            return output_path
            
        except Exception as e:
            # If the AI service fails, fall back to simple extraction.
            service_name = "Ollama" if self.client == "ollama" else "Azure OpenAI"
            if APP_DEBUG:
                print(f"[JSON_GENERATION]   ✗ {service_name} processing failed: {e}")
                print("[JSON_GENERATION]   ⚠ Falling back to simple extraction")
            print(f"Warning: {service_name} processing failed; using simple extraction: {e}")
            return await self._simple_extraction(mineru_json, output_path)
    
    def _load_custom_prompts(self, system_override: Optional[str] = None, user_override: Optional[str] = None) -> tuple[Optional[str], Optional[str]]:
        """
        Load prompts: precedence -- frontend overrides -> .env vars -> AZURE_OPENAI_CUSTOM_PROMPT fallback.
        Note: PROMPT FILE usage removed per configuration; csv_extraction.txt is no longer used.
        """
        # Frontend overrides take highest precedence
        if system_override is not None or user_override is not None:
            return system_override, user_override

        # Read from process environment so updates via load_dotenv(override=True) take effect
        # without requiring an app restart. Store newlines in .env as literal "\\n" sequences.
        system_prompt = os.getenv("AZURE_OPENAI_SYSTEM_PROMPT") or None
        user_prompt = os.getenv("AZURE_OPENAI_USER_PROMPT") or None

        if system_prompt:
            system_prompt = system_prompt.replace("\\\\n", "\n").replace("\\n", "\n")
        if user_prompt:
            user_prompt = user_prompt.replace("\\\\n", "\n").replace("\\n", "\n")

        # Fallback to legacy single PROMPT text if user_prompt not provided
        if not user_prompt and AZURE_OPENAI_CUSTOM_PROMPT:
            user_prompt = AZURE_OPENAI_CUSTOM_PROMPT.replace("\\n", "\n")

        if APP_DEBUG:
            if system_prompt:
                print(f"[JSON_GENERATION]   Using .env AZURE_OPENAI_SYSTEM_PROMPT (length: {len(system_prompt)})")
            if user_prompt:
                print(f"[JSON_GENERATION]   Using .env AZURE_OPENAI_USER_PROMPT (length: {len(user_prompt)})")

        return system_prompt, user_prompt
    
    def _load_custom_prompt(self) -> Optional[str]:
        """
        Load a custom prompt (backward-compatible method).

        This method returns only the user_prompt; system_prompt is retrieved separately.
        """
        _, user_prompt = self._load_custom_prompts()
        return user_prompt
    
    def _extract_content_from_json(self, mineru_json: Dict[str, Any]) -> Dict[str, str]:
        """
        Extract various content fragments from MinerU JSON.

        Returns a dictionary containing different types of extracted content.
        """
        content_dict = {}
        
        # Handle different JSON structures.
        pages = []
        is_content_list_format = False
        
        if isinstance(mineru_json, list):
            # content_list.json format (preferred because it contains the most complete content).
            is_content_list_format = (
                len(mineru_json) > 0 and 
                isinstance(mineru_json[0], dict) and
                "type" in mineru_json[0] and
                "page_idx" in mineru_json[0]
            )
            if is_content_list_format:
                pages = mineru_json
                if APP_DEBUG:
                    print(f"[JSON_GENERATION]   Detected content_list.json format; {len(pages)} items")
        elif isinstance(mineru_json, dict):
            if "document" in mineru_json:
                pages = mineru_json["document"].get("pages", [])
            elif "pages" in mineru_json:
                pages = mineru_json["pages"]
        
        # Extract full content.
        all_text = []
        all_tables = []
        page_texts = {}
        
        for idx, page in enumerate(pages, 1):
            if isinstance(page, dict) and "type" in page:
                # content_list.json format (preferred; most complete content).
                item_type = page.get("type", "")
                if item_type == "text":
                    text = page.get("text", "")
                    if text:
                        all_text.append(text)
                        # If this page doesn't have text yet, set it.
                        page_idx = page.get("page_idx", 0) + 1
                        if f"page_{page_idx}" not in page_texts:
                            page_texts[f"page_{page_idx}"] = text
                        else:
                            page_texts[f"page_{page_idx}"] += "\n" + text
                elif item_type == "table":
                    table_body = page.get("table_body", "")
                    if table_body:
                        # Convert HTML table to a readable text format.
                        table_text = self._parse_html_table_to_text(table_body)
                        all_tables.append(table_text)
                        # Also add table content to text so the AI can extract more easily.
                        all_text.append(table_text)
                        # Update page text.
                        page_idx = page.get("page_idx", 0) + 1
                        if f"page_{page_idx}" not in page_texts:
                            page_texts[f"page_{page_idx}"] = table_text
                        else:
                            page_texts[f"page_{page_idx}"] += "\n\n" + table_text
                elif item_type == "discarded":
                    # Handle content marked as discarded (may still contain useful info).
                    text = page.get("text", "")
                    if text:
                        all_text.append(text)
                        page_idx = page.get("page_idx", 0) + 1
                        if f"page_{page_idx}" not in page_texts:
                            page_texts[f"page_{page_idx}"] = text
                        else:
                            page_texts[f"page_{page_idx}"] += "\n" + text
            elif isinstance(page, dict):
                # Standard page format (vllm-async-engine or standard format).
                text = page.get("text", "")
                elements = page.get("elements", [])
                
                # Extract all content from elements (including text and tables).
                element_texts = []
                element_tables = []
                
                if elements:
                    for element in elements:
                        if isinstance(element, dict):
                            element_type = element.get("type", "").lower()
                            
                            # Extract text content.
                            element_text = element.get("content", "") or element.get("text", "")
                            if element_text:
                                if element_type == "table":
                                    # If it's a table, parse the table content.
                                    table_text = self._parse_html_table_to_text(element_text) if "<table" in element_text.lower() else element_text
                                    element_tables.append(table_text)
                                    element_texts.append(table_text)  # also include in text
                                else:
                                    element_texts.append(element_text)
                
                # Merge page.text with elements content (ensure nothing is lost).
                if element_texts:
                    combined_elements = "\n".join(element_texts)
                    if text:
                        # If both have content, merge them (avoid duplication).
                        if combined_elements not in text:
                            text = text + "\n" + combined_elements
                    else:
                        text = combined_elements
                    if APP_DEBUG:
                        print(f"[JSON_GENERATION]   Extracted text from elements (length: {len(combined_elements)} chars); merged total length: {len(text)} chars")
                
                if text:
                    all_text.append(text)
                    if f"page_{idx}" not in page_texts:
                        page_texts[f"page_{idx}"] = text
                    else:
                        page_texts[f"page_{idx}"] += "\n" + text
                
                # Extract tables (prefer elements, then the tables field).
                if element_tables:
                    for table_text in element_tables:
                        all_tables.append(table_text)
                        if f"page_{idx}" not in page_texts:
                            page_texts[f"page_{idx}"] = table_text
                        else:
                            page_texts[f"page_{idx}"] += "\n\n" + table_text
                
                # Also check the tables field (standard format).
                tables = page.get("tables", [])
                for table in tables:
                    if isinstance(table, dict):
                        # Try extracting table content.
                        table_text = ""
                        if "table_body" in table:
                            table_text = self._parse_html_table_to_text(table.get("table_body", ""))
                        elif "text" in table:
                            table_text = str(table.get("text", ""))
                        else:
                            table_text = str(table)
                        
                        if table_text:
                            all_tables.append(table_text)
                            all_text.append(table_text)  # also include in text
                            if f"page_{idx}" not in page_texts:
                                page_texts[f"page_{idx}"] = table_text
                            else:
                                page_texts[f"page_{idx}"] += "\n\n" + table_text
            elif isinstance(page, (str, int, float)):
                # Simple format.
                text = str(page)
                all_text.append(text)
                page_texts[f"page_{idx}"] = text
        
        # Build content dictionary.
        content_dict["content"] = "\n\n".join(all_text) if all_text else str(mineru_json)
        content_dict["tables"] = "\n\n".join(all_tables) if all_tables else ""
        
        # Add per-page content.
        for page_key, page_text in page_texts.items():
            content_dict[page_key] = page_text
        
        # Limit the length of each content field (configurable via env var; default 500000 chars; 0 means unlimited).
        max_content_length_str = os.getenv("MAX_CONTENT_LENGTH", "500000")
        max_content_length = int(max_content_length_str) if max_content_length_str != "0" else float('inf')
        
        for key in content_dict:
            content_length = len(content_dict[key])
            if content_length > max_content_length:
                if APP_DEBUG:
                    print(f"[JSON_GENERATION]   ⚠ {key} content length ({content_length} chars) exceeds limit ({max_content_length} chars); truncating")
                content_dict[key] = content_dict[key][:max_content_length] + f"...[content truncated, original length: {content_length} chars]"
            elif APP_DEBUG and content_length > 0:
                print(f"[JSON_GENERATION]   ✓ {key} content length: {content_length} chars")
        
        return content_dict
    
    async def _extract_content_from_pdfs(self, pdf_paths: Dict[str, Optional[Path]]) -> Dict[str, Any]:
        """
        Extract content from MinerU-generated PDFs (_layout.pdf and _span.pdf).
        
        Args:
            pdf_paths: Dict containing PDF paths, e.g. {"layout_pdf": Path, "span_pdf": Path}
        
        Returns:
            Extracted content dict in a JSON-like structure.
        """
        from PyPDF2 import PdfReader
        
        all_text = []
        pages_data = []
        
        # Decide which PDFs to use.
        layout_pdf = pdf_paths.get("layout_pdf")
        span_pdf = pdf_paths.get("span_pdf")
        
        pdfs_to_use = []
        if layout_pdf and layout_pdf.exists():
            pdfs_to_use.append(("layout", layout_pdf))
        if span_pdf and span_pdf.exists():
            pdfs_to_use.append(("span", span_pdf))
        
        if not pdfs_to_use:
            if APP_DEBUG:
                print("[JSON_GENERATION] ⚠ No valid PDF files found")
            return {"document": {"pages": [], "total_pages": 0}}
        
        # Extract content from each PDF.
        for pdf_type, pdf_path in pdfs_to_use:
            if APP_DEBUG:
                print(f"[JSON_GENERATION] Extracting content from {pdf_type} PDF: {pdf_path.name}")
            
            try:
                reader = PdfReader(str(pdf_path))
                for page_num, page in enumerate(reader.pages, 1):
                    text = page.extract_text() or ""
                    if text:
                        all_text.append(f"[{pdf_type.upper()}] {text}")
                        pages_data.append({
                            "page_number": page_num,
                            "text": text,
                            "source": pdf_type,
                            "elements": [{
                                "type": "text",
                                "content": text,
                                "bbox": [0, 0, 0, 0]
                            }]
                        })
            except Exception as e:
                if APP_DEBUG:
                    print(f"[JSON_GENERATION] ⚠ Failed to extract content from {pdf_type} PDF: {e}")
        
        if APP_DEBUG:
            print(f"[JSON_GENERATION] Extracted {len(pages_data)} pages from PDFs")
        
        return {
            "document": {
                "pages": pages_data,
                "total_pages": len(pages_data) if pages_data else 0
            },
            "content": "\n\n".join(all_text) if all_text else ""
        }
    
    def _parse_html_table_to_text(self, html_table: str) -> str:
        """
        Convert an HTML table to a readable text format.
        
        Args:
            html_table: Table HTML string
            
        Returns:
            Readable plain-text table
        """
        try:
            # Remove HTML tags while keeping content.
            # First handle <tr> tags (rows).
            text = html_table
            
            # Replace <tr> with newlines.
            text = re.sub(r'<tr[^>]*>', '\n', text)
            text = re.sub(r'</tr>', '', text)
            
            # Handle <td> and <th> tags (cells).
            # Replace with separators (use |).
            text = re.sub(r'<t[dh][^>]*>', '| ', text)
            text = re.sub(r'</t[dh]>', ' |', text)
            
            # Handle colspan/rowspan (simplified: keep content only).
            text = re.sub(r'colspan="\d+"', '', text)
            text = re.sub(r'rowspan="\d+"', '', text)
            
            # Remove remaining HTML tags.
            text = re.sub(r'<[^>]+>', '', text)
            
            # Clean up extra whitespace and separators.
            text = re.sub(r'\s*\|\s*', ' | ', text)  # normalize separator formatting
            text = re.sub(r'\|\s*\|', '|', text)  # remove separators between empty cells
            text = re.sub(r'\n\s*\n', '\n', text)  # remove extra blank lines
            text = text.strip()
            
            # If conversion fails or result is empty, return original HTML (at least the AI can see it).
            if not text or len(text) < 10:
                # Fallback: strip all tags directly.
                text = re.sub(r'<[^>]+>', ' | ', html_table)
                text = re.sub(r'\s*\|\s*', ' | ', text)
                text = re.sub(r'\|\s*\|', '|', text)
                text = text.strip()
            
            return text if text else html_table
            
        except Exception as e:
            if APP_DEBUG:
                print(f"[JSON_GENERATION]   ⚠ Error while parsing HTML table: {e}")
            # If parsing fails, return the original HTML.
            return html_table
    
    def _replace_placeholders(self, template: str, content_dict: Dict[str, str]) -> str:
        """
        Replace placeholders in the template.

        Supports:
        - {content}: full content
        - {tables}: all tables
        - {page_1}, {page_2}, ...: specific page content
        - other custom placeholders (if present in content_dict)
        - semantic placeholders (e.g. {invoice date}, {amount}): keep as-is if missing so the AI can interpret
        """
        # Find all placeholders {xxx}.
        placeholders = re.findall(r'\{([^}]+)\}', template)
        
        if APP_DEBUG and placeholders:
            print(f"[JSON_GENERATION]   Found placeholders: {placeholders}")
        
        result = template
        replaced_placeholders = []
        semantic_placeholders = []
        
        for placeholder in placeholders:
            # Check if this is a built-in placeholder in the content dictionary.
            if placeholder in content_dict:
                replacement = content_dict[placeholder]
                result = result.replace(f"{{{placeholder}}}", replacement)
                replaced_placeholders.append(placeholder)
                if APP_DEBUG:
                    print(f"[JSON_GENERATION]   ✓ Replaced {{{placeholder}}} (length: {len(replacement)} chars)")
            else:
                # Semantic placeholder (e.g., {invoice date}, {amount}).
                # Keep as-is so the model can infer/extract from {content}.
                semantic_placeholders.append(placeholder)
                if APP_DEBUG:
                    print(f"[JSON_GENERATION]   ℹ Keeping semantic placeholder {{{placeholder}}}; AI will extract from content")
        
        if APP_DEBUG:
            if replaced_placeholders:
                print(f"[JSON_GENERATION]   Replaced placeholders: {replaced_placeholders}")
            if semantic_placeholders:
                print(f"[JSON_GENERATION]   Semantic placeholders (interpreted by AI): {semantic_placeholders}")
        
        return result
    
    def _create_extraction_prompt_from_template(
        self,
        mineru_json: Dict[str, Any],
        template: str
    ) -> str:
        """
        Create an extraction prompt from a template (supports placeholder replacement).
        
        Args:
            mineru_json: MinerU JSON data
            template: Prompt template
        
        Returns:
            Prompt with placeholders replaced
        """
        # Extract content for placeholder replacement.
        content_dict = self._extract_content_from_json(mineru_json)
        
        # Replace placeholders.
        prompt = self._replace_placeholders(template, content_dict)
        
        # If the template has no placeholders, append full content.
        if not re.search(r'\{[^}]+\}', template):
            if APP_DEBUG:
                print("[JSON_GENERATION]   No placeholders in prompt; appending full content")
            prompt = f"{template}\n\nPDF content:\n{content_dict.get('content', '')}"
        
        return prompt
    
    def _create_extraction_prompt(self, mineru_json: Dict[str, Any]) -> str:
        """Build a data extraction prompt."""
        # Extract content fragments.
        content_dict = self._extract_content_from_json(mineru_json)
        
        # Try loading a custom prompt.
        custom_prompt_template = self._load_custom_prompt()
        
        if custom_prompt_template:
            # Use custom prompt and replace placeholders.
            prompt = self._replace_placeholders(custom_prompt_template, content_dict)
            
            # If the template has no placeholders, append full content.
            if not re.search(r'\{[^}]+\}', custom_prompt_template):
                if APP_DEBUG:
                    print("[JSON_GENERATION]   No placeholders in prompt; appending full content")
                prompt = f"{custom_prompt_template}\n\nPDF content:\n{content_dict.get('content', '')}"
        else:
            # Default prompt.
            prompt = f"""
Extract structured data from the following PDF content and output JSON.

PDF content:
{content_dict.get('content', '')}

Please analyze the content and extract:
1. Table data (if present)
2. Key fields (e.g., dates, amounts, names)
3. List items

Return ONLY valid JSON (no markdown/code fences). If you cannot extract structured data, return a best-effort JSON with the main text organized into a table-like structure.
"""
        
        return prompt
    
    def _extract_csv_from_response(self, response: str) -> str:
        """
        Extract CSV content from an AI response.

        Supports extracting from markdown code blocks (e.g. ```csv or ```).
        """
        response = response.strip()
        
        # Try extracting from markdown code blocks.
        # Match ```csv ... ``` or ``` ... ```
        csv_patterns = [
            r'```csv\s*\n(.*?)\n```',  # ```csv ... ```
            r'```\s*\n(.*?)\n```',      # ``` ... ```
            r'```csv\s*(.*?)\n```',     # ```csv ... ``` (single-line)
            r'```\s*(.*?)\n```',        # ``` ... ``` (single-line)
        ]
        
        for pattern in csv_patterns:
            match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
            if match:
                extracted = match.group(1).strip()
                if self._is_csv_format(extracted):
                    if APP_DEBUG:
                        print("[JSON_GENERATION]   ✓ Extracted table content from markdown code block")
                    return extracted
        
        # If no code block, check whether the whole response is CSV.
        if self._is_csv_format(response):
            return response
        
        # Try finding a CSV-like section (comma-separated table).
        lines = response.split('\n')
        csv_lines = []
        for line in lines:
            # If the line contains commas and looks like CSV (>= 2 fields).
            if ',' in line and len(line.split(',')) >= 2:
                csv_lines.append(line)
        
        if len(csv_lines) >= 2:  # at least header + one data row
            extracted = '\n'.join(csv_lines)
            if APP_DEBUG:
                print(f"[JSON_GENERATION]   ✓ Extracted {len(csv_lines)} CSV-like lines from response")
            return extracted
        
        return ""

    def _extract_codeblock_from_response(self, response: str, lang_hint: Optional[str] = None) -> Optional[str]:
        """
        Extract the contents of a code block from a response (e.g. ```json ... ``` or unlabeled ``` ... ```).

        If lang_hint is provided, prefer matching that language first.
        """
        if not response:
            return None
        # Prefer matching a language-specific code block.
        if lang_hint:
            pattern = rf'```\s*{re.escape(lang_hint)}\s*\n(.*?)\n```'
            match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
            if match:
                return match.group(1).strip()

        # Match any code block.
        match = re.search(r'```\s*\n(.*?)\n```', response, re.DOTALL)
        if match:
            return match.group(1).strip()

        # Try matching the single-line form: ```json {...} ```
        match = re.search(r'```\s*json\s*(\{.*?\})\s*```', response, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()

        return None
    
    def _is_csv_format(self, content: str) -> bool:
        """Check whether the content looks like CSV."""
        if not content or not content.strip():
            return False
        
        lines = content.strip().split("\n")
        if len(lines) < 2:
            return False
        
        # Check the first line (header) contains commas.
        first_line = lines[0].strip()
        if not first_line or ',' not in first_line:
            return False
        
        # Check there are at least 2 fields.
        fields = first_line.split(',')
        if len(fields) < 2:
            return False
        
        return True
    
    def _standardize_csv_format(self, csv_content: str) -> str:
        """
        Standardize CSV formatting for consistency:
        - Normalize field counts
        - Trim extra whitespace
        - Ensure correct quote handling
        - Remove empty lines
        """
        lines = csv_content.strip().split('\n')
        if not lines:
            return csv_content
        
        # Parse CSV (handle commas inside quotes).
        import csv
        from io import StringIO
        
        try:
            reader = csv.reader(StringIO(csv_content))
            rows = list(reader)
            
            if not rows:
                return csv_content
            
            # Determine the number of fields from the header row.
            num_fields = len(rows[0])
            
            # Standardize each row to ensure consistent field count.
            standardized_rows = []
            for row in rows:
                # Ensure field count matches.
                while len(row) < num_fields:
                    row.append('')  # fill missing fields
                if len(row) > num_fields:
                    row = row[:num_fields]  # truncate extra fields
                
                # Clean each field (strip outer whitespace).
                standardized_row = [field.strip() if isinstance(field, str) else str(field).strip() for field in row]
                standardized_rows.append(standardized_row)
            
            # Write back standardized CSV.
            output = StringIO()
            writer = csv.writer(output, quoting=csv.QUOTE_MINIMAL)
            for row in standardized_rows:
                writer.writerow(row)
            
            result = output.getvalue().strip()
            if APP_DEBUG:
                print(f"[JSON_GENERATION]   ✓ Table standardized: {len(standardized_rows)} rows, {num_fields} columns")
            return result
            
        except Exception as e:
            if APP_DEBUG:
                print(f"[JSON_GENERATION]   ⚠ Table standardization failed; using raw content: {e}")
            return csv_content
    
    def _standardize_dataframe_to_csv(self, df: pd.DataFrame) -> str:
        """
        Standardize a DataFrame into a CSV-formatted string.
        """
        if df.empty:
            return ""
        
        # Clean column names (trim whitespace).
        df.columns = [str(col).strip() for col in df.columns]
        
        # Clean data (convert NaN to empty string).
        df = df.fillna('')
        
        # Convert to CSV string.
        from io import StringIO
        output = StringIO()
        df.to_csv(output, index=False, encoding='utf-8-sig')
        csv_content = output.getvalue()
        
        # Strip trailing newlines (pandas adds them).
        return csv_content.rstrip('\n\r')
    
    def _parse_ai_response_to_dataframe(
        self,
        ai_response: str,
        mineru_json: Dict[str, Any],
        output_format: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Parse an AI response into a DataFrame.

        Prefer data from the AI response over mineru_json.
        """
        try:
            # Method 1: try parsing as JSON.
            if ai_response.strip().startswith("{"):
                try:
                    data = json.loads(ai_response)
                    if isinstance(data, list):
                        if APP_DEBUG:
                            print(f"[JSON_GENERATION]   ✓ Parsed JSON list from AI response ({len(data)} items)")
                        return pd.DataFrame(data)
                    elif isinstance(data, dict):
                        if "data" in data:
                            if APP_DEBUG:
                                print("[JSON_GENERATION]   ✓ Parsed JSON dict from AI response (data field)")
                            return pd.DataFrame(data["data"])
                        else:
                            # Convert whole dict into a single-row DataFrame.
                            if APP_DEBUG:
                                print("[JSON_GENERATION]   ✓ Parsed JSON dict from AI response (single row)")
                            return pd.DataFrame([data])
                except json.JSONDecodeError:
                    if APP_DEBUG:
                        print("[JSON_GENERATION]   ⚠ AI response looks like JSON but failed to parse")
            
            # Method 2: extract a table structure (only when CSV is needed).
            # Look for lines with multiple columns.
            lines = ai_response.strip().split('\n')
            table_lines = []
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                # If the line contains multiple columns (comma/tab/pipe separated).
                if ',' in line or '\t' in line or '|' in line:
                    # Normalize pipe-separated markdown tables.
                    if '|' in line:
                        line = line.strip('|')
                        parts = [p.strip() for p in line.split('|')]
                        line = ','.join(parts)
                    table_lines.append(line)
            
            if (output_format or "").lower() == "csv" and len(table_lines) >= 2:  # header + at least one row
                if APP_DEBUG:
                    print(f"[JSON_GENERATION]   ✓ Extracted table structure from AI response ({len(table_lines)} lines)")
                # Try parsing as CSV.
                try:
                    from io import StringIO
                    df = pd.read_csv(StringIO('\n'.join(table_lines)))
                    return df
                except Exception as e:
                    if APP_DEBUG:
                        print(f"[JSON_GENERATION]   ⚠ Failed to parse as a standard table: {e}")
            
            # Method 3: last resort.
            if APP_DEBUG:
                print("[JSON_GENERATION]   ⚠ Could not extract structured data from AI response; using raw response")
            
            # Return raw AI response as a single field.
            return pd.DataFrame([{"AI response": ai_response}])
            
        except Exception as e:
            if APP_DEBUG:
                print(f"[JSON_GENERATION]   ✗ Error while parsing AI response: {e}")
                print("[JSON_GENERATION]   ⚠ Using raw AI response")
            return pd.DataFrame([{"AI response": ai_response}])
    
    async def _program_csv_extraction(
        self, 
        mineru_json: Dict[str, Any], 
        output_path: Path
    ) -> Path:
        """
        Programmatic extraction: flatten MinerU JSON into a stable output.

        Default output is JSON (when output_path suffix is .json or anything other than .csv).
        If output_path suffix is .csv, output CSV for backward compatibility.
        """
        if APP_DEBUG:
            print("[JSON_GENERATION] Using program_csv extraction (programmatic flattening)")
            print("[JSON_GENERATION]   Output is generated programmatically (no AI; consistent format)")
        
        rows = []
        
        output_is_csv = (output_path.suffix or "").lower() == ".csv"

        # Check whether this is content_list.json format (array).
        if isinstance(mineru_json, list):
            if APP_DEBUG:
                print(f"[PROGRAM_EXTRACTION] Detected content_list.json format; {len(mineru_json)} items")
            
            # Collect all possible field names from all items.
            all_field_names = set()
            for item in mineru_json:
                if isinstance(item, dict):
                    all_field_names.update(item.keys())
            
            # Sort field names to keep ordering stable.
            field_names = sorted(list(all_field_names))
            
            if APP_DEBUG:
                print(f"[PROGRAM_EXTRACTION] Detected {len(field_names)} fields: {', '.join(field_names)}")
            
            # Flatten each item into one row.
            for idx, item in enumerate(mineru_json):
                if isinstance(item, dict):
                    row = {}
                    for field in field_names:
                        value = item.get(field, "")
                        
                        # Handle special fields.
                        if field == "bbox" and isinstance(value, list):
                            # Convert bbox array to string.
                            value = ",".join(str(v) for v in value)
                        elif field == "table_body" and isinstance(value, str):
                            # Convert HTML table to text (simplified).
                            table_text = self._parse_html_table_to_text(value)
                            row["table_body_text"] = table_text  # add an extra text version
                            value = value[:500] if len(value) > 500 else value  # limit length
                        elif field == "table_caption" or field == "table_footnote":
                            # Convert arrays to string.
                            if isinstance(value, list):
                                value = "; ".join(str(v) for v in value)
                        elif isinstance(value, (list, dict)):
                            # Convert complex objects to JSON strings.
                            value = json.dumps(value, ensure_ascii=False)
                        
                        row[field] = str(value) if value is not None else ""
                    
                    rows.append(row)
        elif isinstance(mineru_json, dict):
            # If it's a dict, try extracting standard pages.
            if "document" in mineru_json:
                pages = mineru_json["document"].get("pages", [])
                if APP_DEBUG:
                    print(f"[PROGRAM_EXTRACTION] Detected standard format; {len(pages)} pages")
                
                # Convert standard format to a flattened format.
                all_field_names = set()
                for page in pages:
                    if isinstance(page, dict):
                        all_field_names.update(page.keys())
                        # Also include element fields.
                        if "elements" in page:
                            for element in page.get("elements", []):
                                if isinstance(element, dict):
                                    all_field_names.update([f"element_{k}" for k in element.keys()])
                
                field_names = sorted(list(all_field_names))
                
                for page_idx, page in enumerate(pages):
                    if isinstance(page, dict):
                        row = {"page_number": page_idx + 1}
                        for field in field_names:
                            if field.startswith("element_"):
                                continue  # skip element_-prefixed fields
                            value = page.get(field, "")
                            if isinstance(value, (list, dict)):
                                value = json.dumps(value, ensure_ascii=False)
                            row[field] = str(value) if value is not None else ""
                        rows.append(row)
            else:
                # Flatten dict directly.
                if APP_DEBUG:
                    print("[PROGRAM_EXTRACTION] Detected dict format; flattening")
                rows.append(self._flatten_dict(mineru_json))
        else:
            # Other formats: create a single row.
            if APP_DEBUG:
                print("[PROGRAM_EXTRACTION] Unknown format; creating single row")
            rows.append({"content": str(mineru_json)})
        
        if not rows:
            if APP_DEBUG:
                print("[PROGRAM_EXTRACTION] ⚠ No data extracted")
            # Create empty output.
            if output_is_csv:
                with open(output_path, "w", encoding="utf-8-sig") as f:
                    f.write("")
            else:
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump([], f, ensure_ascii=False, indent=2)
            return output_path

        if output_is_csv:
            # Convert to DataFrame and save as CSV (backward compatible).
            df = pd.DataFrame(rows)
            standardized_csv = self._standardize_dataframe_to_csv(df)
            with open(output_path, "w", encoding="utf-8-sig") as f:
                f.write(standardized_csv)
            if APP_DEBUG:
                print(f"[PROGRAM_EXTRACTION] ✓ Table generated: {len(rows)} rows, {len(df.columns)} columns")
                print(f"[PROGRAM_EXTRACTION]   Columns: {', '.join(df.columns.tolist()[:10])}{'...' if len(df.columns) > 10 else ''}")
            return output_path

        # Default JSON output: write rows directly.
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)
        if APP_DEBUG:
            sample_keys = []
            if rows and isinstance(rows[0], dict):
                sample_keys = list(rows[0].keys())
            print(f"[PROGRAM_EXTRACTION] ✓ JSON generated: {len(rows)} records")
            if sample_keys:
                print(f"[PROGRAM_EXTRACTION]   Fields: {', '.join(sample_keys[:10])}{'...' if len(sample_keys) > 10 else ''}")
        return output_path
    
    def _flatten_dict(self, d: Dict[str, Any], parent_key: str = "", sep: str = "_") -> Dict[str, Any]:
        """
        Recursively flatten a dictionary.
        """
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            elif isinstance(v, list):
                # Convert lists to JSON strings.
                items.append((new_key, json.dumps(v, ensure_ascii=False)))
            else:
                items.append((new_key, str(v) if v is not None else ""))
        return dict(items)
    
    async def _simple_extraction(
        self, 
        mineru_json: Dict[str, Any], 
        output_path: Path
    ) -> Path:
        """Simple extraction method (used when AI is unavailable)."""
        if APP_DEBUG:
            print("[JSON_GENERATION] Using simple extraction (no AI)")
            print("[JSON_GENERATION]   Output is generated by simple extraction (no AI)")
        
        rows = []
        
        output_is_csv = (output_path.suffix or "").lower() == ".csv"

        # DEBUG output: show JSON structure.
        if APP_DEBUG:
            print(f"[SIMPLE_EXTRACTION] JSON type: {type(mineru_json)}")
            if isinstance(mineru_json, dict):
                keys = list(mineru_json.keys())[:10]
                print(f"[SIMPLE_EXTRACTION] Top-level keys: {keys}")
            elif isinstance(mineru_json, list):
                print(f"[SIMPLE_EXTRACTION] JSON is a list; length: {len(mineru_json)}")
        
        try:
            # Handle different JSON structures. Possible cases:
            # 1. content_list.json format: [{"type": "text/table", "text": "...", "page_idx": 0, ...}, ...]
            # 2. {"document": {"pages": [...]}}
            # 3. {"pages": [...]}
            # 4. a raw list [...]
            
            # Check whether this is content_list.json format (list; each item has type and page_idx).
            is_content_list_format = (
                isinstance(mineru_json, list) and 
                len(mineru_json) > 0 and 
                isinstance(mineru_json[0], dict) and
                "type" in mineru_json[0] and
                "page_idx" in mineru_json[0]
            )
            
            if is_content_list_format:
                # Handle content_list.json format.
                if APP_DEBUG:
                    print(f"[SIMPLE_EXTRACTION] Detected content_list.json format; {len(mineru_json)} items")
                
                for item in mineru_json:
                    if not isinstance(item, dict):
                        continue
                    
                    item_type = item.get("type", "unknown")
                    page_idx = item.get("page_idx", 0)
                    
                    if item_type == "text":
                        text = item.get("text", "")
                        if text:
                            rows.append({
                                "page": page_idx + 1,  # convert 0-based page_idx to 1-based
                                "type": "text",
                                "content": text,
                                "font_size": "",
                                "font_family": ""
                            })
                    elif item_type == "table":
                        table_body = item.get("table_body", "")
                        if table_body:
                            rows.append({
                                "page": page_idx + 1,
                                "type": "table",
                                "content": table_body,  # HTML table
                                "font_size": "",
                                "font_family": ""
                            })
                    # Skip "discarded" items.
                
                if rows:
                    if APP_DEBUG:
                        print(f"[SIMPLE_EXTRACTION] Extracted {len(rows)} records from content_list.json")

                    if output_is_csv:
                        df = pd.DataFrame(rows)
                        df.to_csv(output_path, index=False, encoding="utf-8-sig")
                    else:
                        with open(output_path, "w", encoding="utf-8") as f:
                            json.dump(rows, f, ensure_ascii=False, indent=2)
                    return output_path
            
            # Handle other formats.
            pages = []
            if isinstance(mineru_json, dict):
                if "document" in mineru_json:
                    pages = mineru_json["document"].get("pages", [])
                    if APP_DEBUG:
                        print(f"[SIMPLE_EXTRACTION] Found {len(pages)} pages in document.pages")
                elif "pages" in mineru_json:
                    pages = mineru_json["pages"]
                    if APP_DEBUG:
                        print(f"[SIMPLE_EXTRACTION] Found {len(pages)} pages in pages")
            elif isinstance(mineru_json, list):
                pages = mineru_json
                if APP_DEBUG:
                    print(f"[SIMPLE_EXTRACTION] JSON is a page list; {len(pages)} pages")
            
            if not pages:
                # If no pages found, extract top-level text at minimum.
                if isinstance(mineru_json, dict):
                    text_content = str(mineru_json)
                    rows.append({
                        "page": 0,
                        "type": "text",
                        "content": text_content,
                        "font_size": "",
                        "font_family": ""
                    })
                else:
                    rows.append({
                        "page": 0,
                        "type": "text",
                        "content": str(mineru_json),
                        "font_size": "",
                        "font_family": ""
                    })
            else:
                # Process each page.
                for page_idx, page in enumerate(pages, 1):
                    # Ensure page is a dict.
                    if not isinstance(page, dict):
                        # If not a dict, convert simple types or skip.
                        if isinstance(page, (str, int, float)):
                            rows.append({
                                "page": page_idx,
                                "type": "text",
                                "content": str(page),
                                "font_size": "",
                                "font_family": ""
                            })
                        continue
                    
                    page_num = page.get("page_number", page_idx)
                    text = page.get("text", "")
                    
                    # If page text exists, add a text row first.
                    if text:
                        rows.append({
                            "page": page_num,
                            "type": "text",
                            "content": text,
                            "font_size": "",
                            "font_family": ""
                        })
                    
                    # Extract elements (ensure elements is a list).
                    elements = page.get("elements", [])
                    if isinstance(elements, list):
                        for element in elements:
                            # Ensure element is a dict.
                            if isinstance(element, dict):
                                element_type = element.get("type", "text")
                                if element_type == "text" or "content" in element:
                                    rows.append({
                                        "page": page_num,
                                        "type": "element",
                                        "content": element.get("content", element.get("text", "")),
                                        "font_size": str(element.get("font_size", "")),
                                        "font_family": element.get("font_family", "")
                                    })
                            elif isinstance(element, (str, int, float)):
                                # If element is a simple type, add directly.
                                rows.append({
                                    "page": page_num,
                                    "type": "element",
                                    "content": str(element),
                                    "font_size": "",
                                    "font_family": ""
                                })
                    
                    # Extract tables.
                    tables = page.get("tables", [])
                    if isinstance(tables, list):
                        for table in tables:
                            if isinstance(table, dict):
                                rows.append({
                                    "page": page_num,
                                    "type": "table",
                                    "content": json.dumps(table, ensure_ascii=False),
                                    "font_size": "",
                                    "font_family": ""
                                })
                            elif isinstance(table, (str, list)):
                                rows.append({
                                    "page": page_num,
                                    "type": "table",
                                    "content": json.dumps(table, ensure_ascii=False),
                                    "font_size": "",
                                    "font_family": ""
                                })
        except Exception as e:
            # If processing fails, save a minimal error record.
            if APP_DEBUG:
                print(f"[SIMPLE_EXTRACTION] ⚠ Error while extracting data: {e}")
            rows.append({
                "page": 0,
                "type": "error",
                "content": f"Extraction failed: {str(e)}",
                "font_size": "",
                "font_family": ""
            })
        
        # If no rows were extracted, add an explicit placeholder row.
        if not rows:
            rows.append({
                "page": 0,
                "type": "no_data",
                "content": "No data could be extracted from JSON",
                "font_size": "",
                "font_family": ""
            })
        
        if APP_DEBUG:
            print(f"[SIMPLE_EXTRACTION] Extracted {len(rows)} records")

        if output_is_csv:
            df = pd.DataFrame(rows)
            df.to_csv(output_path, index=False, encoding="utf-8-sig")
        else:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(rows, f, ensure_ascii=False, indent=2)
        
        if APP_DEBUG:
            print(f"[JSON_GENERATION]   ✓ Output file generated: {output_path}")
            print("[JSON_GENERATION]   ⚠ Output is generated by simple extraction (no AI)")
        
        return output_path
    


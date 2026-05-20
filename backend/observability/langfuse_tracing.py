# backend/observability/langfuse_tracing.py
#
# Instantiating Langfuse() registers the global OpenTelemetry tracer so that
# @observe() (imported from langfuse.decorators) picks up the correct project/host.
# Use LANGFUSE_BASE_URL (not LANGFUSE_HOST) per project convention.
import os
from dotenv import load_dotenv
from langfuse import Langfuse

load_dotenv()

langfuse_client = Langfuse(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    base_url=os.getenv("LANGFUSE_BASE_URL"),
)

"""
Bootstrap Script
================

Startup script for the Sentinel RAG Agent.
Loads configuration and initializes OpenTelemetry.
"""

import logging
import os
from pathlib import Path

import yaml
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "sentinel_config.yaml"


def load_config(config_path: Path | None = None) -> dict:
    if config_path is None:
        config_path = CONFIG_PATH
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    logger.info("Loading configuration from %s", config_path)
    
    with open(config_path) as f:
        raw_config = yaml.safe_load(f)
    
    raw_config = _expand_env_vars(raw_config)
    return raw_config


def _expand_env_vars(obj):
    if isinstance(obj, str):
        if obj.startswith("${") and obj.endswith("}"):
            var_name = obj[2:-1]
            return os.environ.get(var_name, obj)
        return obj
    elif isinstance(obj, dict):
        return {k: _expand_env_vars(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_expand_env_vars(item) for item in obj]
    return obj


def setup_telemetry(service_name: str = "sentinel-rag-agent", service_version: str = "1.0.0"):
    otel_enabled = os.environ.get("OTEL_ENABLED", "true").lower() == "true"
    
    if not otel_enabled:
        logger.info("OpenTelemetry disabled")
        return
    
    otlp_endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")
    logger.info("Setting up OpenTelemetry: endpoint=%s", otlp_endpoint)
    
    resource = Resource.create({
        "service.name": service_name,
        "service.version": service_version,
        "deployment.environment": os.environ.get("ENVIRONMENT", "development"),
    })
    
    tracer_provider = TracerProvider(resource=resource)
    
    try:
        otlp_exporter = OTLPSpanExporter(endpoint=otlp_endpoint, insecure=True)
        span_processor = BatchSpanProcessor(otlp_exporter)
        tracer_provider.add_span_processor(span_processor)
    except Exception as e:
        logger.warning("Failed to setup OTLP exporter: %s", str(e))
    
    trace.set_tracer_provider(tracer_provider)
    HTTPXClientInstrumentor().instrument()
    logger.info("OpenTelemetry initialized successfully")


def bootstrap(config_path: Path | None = None) -> dict:
    logger.info("Starting Sentinel RAG Agent bootstrap...")
    
    config = load_config(config_path)
    setup_telemetry(
        service_name=config.get("service_name", "sentinel-rag-agent"),
        service_version=config.get("version", "1.0.0"),
    )
    
    logger.info("Bootstrap completed successfully")
    return config


def setup_logging(level: str = "INFO"):
    log_format = os.environ.get("LOG_FORMAT", "text")
    
    if log_format == "json":
        import json
        
        class JsonFormatter(logging.Formatter):
            def format(self, record: logging.LogRecord) -> str:
                log_data = {
                    "timestamp": self.formatTime(record),
                    "level": record.levelname,
                    "logger": record.name,
                    "message": record.getMessage(),
                }
                if record.exc_info:
                    log_data["exception"] = self.formatException(record.exc_info)
                return json.dumps(log_data)
        
        handler = logging.StreamHandler()
        handler.setFormatter(JsonFormatter())
    else:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))
    root_logger.addHandler(handler)
    
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("opentelemetry").setLevel(logging.WARNING)

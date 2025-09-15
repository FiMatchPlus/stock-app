"""로깅 설정"""

import structlog
from datetime import datetime

def get_logger(name: str = "stock_server"):
    """구조화된 로거 생성"""
    
    # Structlog 설정
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    return structlog.get_logger(name)

def log_api_request(logger, method: str, url: str, **kwargs):
    """API 요청 로그를 기록하는 헬퍼 함수"""
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.info(
        "API Request",
        method=method,
        url=url,
        timestamp=current_time,
        **kwargs
    )




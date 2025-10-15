"""로깅 설정"""

import structlog
import logging
import sys
import json
from datetime import datetime

def pretty_format(logger, method_name, event_dict):
    """로그를 보기 좋게 포맷팅"""
    
    event = event_dict.pop('event', '')
    
    result_parts = [event]
    
    priority_keys = ['benchmark_code', 'start_date', 'end_date', 'data_points', 
                     'error', 'exception', 'execution_time', 'portfolio_id']
    
    for key in priority_keys:
        if key in event_dict:
            value = event_dict.pop(key)
            result_parts.append(f"    {key}: {value}")
    
    for key, value in event_dict.items():
        if isinstance(value, (dict, list)) and len(str(value)) > 80:
            if isinstance(value, dict):
                formatted = json.dumps(value, indent=2, ensure_ascii=False)
            else:
                formatted = str(value)
            
            result_parts.append(f"    {key}:")
            for line in formatted.split('\n'):
                result_parts.append(f"      {line}")
        else:
            result_parts.append(f"    {key}: {value}")
    
    return '\n'.join(result_parts)

def get_logger(name: str = "stock_server"):
    """구조화된 로거 생성"""
    
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(message)s')
    handler.setFormatter(formatter)
    
    logging.basicConfig(
        level=logging.INFO,
        handlers=[handler]
    )
    
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            pretty_format,
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
        "API 요청",
        method=method,
        url=url,
        timestamp=current_time,
        **kwargs
    )




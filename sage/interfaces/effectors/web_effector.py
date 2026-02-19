"""
WebEffector — HTTP requests and API calls as Effects.

Handles EffectType.API_CALL and EffectType.WEB with domain allowlisting
and rate limiting.
"""

import time
from typing import Dict, Any, Tuple, List
from urllib.parse import urlparse

from ..base_effector import BaseEffector, EffectorCommand, EffectorResult, EffectorStatus


class WebEffector(BaseEffector):
    """
    Handles EffectType.API_CALL and EffectType.WEB.

    Actions: get, post, put, delete, head
    Target: URL (via command.metadata['target'])
    Safety: domain allowlist, rate limiting, response size cap.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.allowed_domains: List[str] = config.get('allowed_domains', [])
        self.rate_limit_per_second: float = config.get('rate_limit', 10.0)
        self.max_response_size: int = config.get('max_response_size', 10 * 1024 * 1024)
        self._request_times: List[float] = []

    def execute(self, command: EffectorCommand) -> EffectorResult:
        start_time = time.time()

        error = self._check_enabled()
        if error:
            return error

        is_valid, message = self.validate_command(command)
        if not is_valid:
            result = EffectorResult(
                effector_id=self.effector_id,
                status=EffectorStatus.INVALID_COMMAND,
                message=message,
            )
            self._update_stats(result)
            return result

        # Rate limit check
        now = time.time()
        self._request_times = [t for t in self._request_times if now - t < 1.0]
        if len(self._request_times) >= self.rate_limit_per_second:
            result = EffectorResult(
                effector_id=self.effector_id,
                status=EffectorStatus.FAILED,
                message="Rate limit exceeded",
                execution_time=time.time() - start_time,
            )
            self._update_stats(result)
            return result

        url = command.metadata.get('target', command.parameters.get('url', ''))
        method = command.action.upper()

        try:
            import urllib.request
            import json

            req = urllib.request.Request(url, method=method)

            # Add headers
            headers = command.parameters.get('headers', {})
            for key, val in headers.items():
                req.add_header(key, val)

            # Add body for POST/PUT
            body = command.parameters.get('body', None)
            data = None
            if body and method in ('POST', 'PUT'):
                if isinstance(body, dict):
                    data = json.dumps(body).encode('utf-8')
                    req.add_header('Content-Type', 'application/json')
                elif isinstance(body, str):
                    data = body.encode('utf-8')

            with urllib.request.urlopen(req, data=data,
                                       timeout=command.timeout) as resp:
                response_body = resp.read(self.max_response_size)
                status_code = resp.status

            self._request_times.append(now)

            result = EffectorResult(
                effector_id=self.effector_id,
                status=EffectorStatus.SUCCESS,
                message=f"{method} {url} -> {status_code}",
                execution_time=time.time() - start_time,
                metadata={
                    'status_code': status_code,
                    'response_size': len(response_body),
                    'url': url,
                },
            )

        except Exception as e:
            result = EffectorResult(
                effector_id=self.effector_id,
                status=EffectorStatus.FAILED,
                message=f"{method} {url} failed: {str(e)}",
                execution_time=time.time() - start_time,
            )

        self._update_stats(result)
        return result

    def validate_command(self, command: EffectorCommand) -> Tuple[bool, str]:
        if command.action not in ('get', 'post', 'put', 'delete', 'head'):
            return False, f"Invalid action: {command.action}"

        url = command.metadata.get('target', command.parameters.get('url', ''))
        if not url:
            return False, "Missing URL"

        # Domain allowlist
        if self.allowed_domains:
            parsed = urlparse(url)
            domain = parsed.netloc or parsed.hostname or ''
            if not any(domain.endswith(d) for d in self.allowed_domains):
                return False, f"Domain '{domain}' not in allowed list"

        return True, ""

    def is_available(self) -> bool:
        return True

    def get_info(self) -> Dict[str, Any]:
        return {
            'effector_id': self.effector_id,
            'effector_type': 'web',
            'supported_actions': ['get', 'post', 'put', 'delete', 'head'],
            'allowed_domains': self.allowed_domains,
            'rate_limit': self.rate_limit_per_second,
            'max_response_size': self.max_response_size,
        }

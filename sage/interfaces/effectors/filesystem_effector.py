"""
FileSystemEffector — read/write/delete files as Effects.

Handles EffectType.FILE_IO with sandboxed path validation.
"""

import os
import time
import fnmatch
from pathlib import Path
from typing import Dict, Any, Tuple, List

from ..base_effector import BaseEffector, EffectorCommand, EffectorResult, EffectorStatus


class FileSystemEffector(BaseEffector):
    """
    Handles EffectType.FILE_IO.

    Actions: read, write, append, delete, list
    Target: file path (via command.metadata['target'])
    Safety: sandboxed to allowed_paths, deny_patterns for sensitive files.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.allowed_paths: List[str] = config.get('allowed_paths', [])
        self.deny_patterns: List[str] = config.get(
            'deny_patterns',
            ['*.env', '*password*', '*credential*', '*secret*', '*.key']
        )

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

        safety = self._check_safety(command)
        if safety:
            return safety

        target = command.metadata.get('target', command.parameters.get('path', ''))

        try:
            if command.action == 'read':
                content = Path(target).read_text(
                    encoding=command.parameters.get('encoding', 'utf-8')
                )
                result = EffectorResult(
                    effector_id=self.effector_id,
                    status=EffectorStatus.SUCCESS,
                    message=f"Read {len(content)} chars from {target}",
                    execution_time=time.time() - start_time,
                    metadata={'content': content, 'size': len(content)},
                )

            elif command.action == 'write':
                content = command.parameters.get('content', '')
                Path(target).parent.mkdir(parents=True, exist_ok=True)
                Path(target).write_text(
                    content,
                    encoding=command.parameters.get('encoding', 'utf-8')
                )
                result = EffectorResult(
                    effector_id=self.effector_id,
                    status=EffectorStatus.SUCCESS,
                    message=f"Wrote {len(content)} chars to {target}",
                    execution_time=time.time() - start_time,
                    metadata={'bytes_written': len(content)},
                )

            elif command.action == 'append':
                content = command.parameters.get('content', '')
                with open(target, 'a',
                          encoding=command.parameters.get('encoding', 'utf-8')) as f:
                    f.write(content)
                result = EffectorResult(
                    effector_id=self.effector_id,
                    status=EffectorStatus.SUCCESS,
                    message=f"Appended {len(content)} chars to {target}",
                    execution_time=time.time() - start_time,
                    metadata={'bytes_appended': len(content)},
                )

            elif command.action == 'delete':
                Path(target).unlink(missing_ok=True)
                result = EffectorResult(
                    effector_id=self.effector_id,
                    status=EffectorStatus.SUCCESS,
                    message=f"Deleted {target}",
                    execution_time=time.time() - start_time,
                )

            elif command.action == 'list':
                entries = [str(p) for p in Path(target).iterdir()]
                result = EffectorResult(
                    effector_id=self.effector_id,
                    status=EffectorStatus.SUCCESS,
                    message=f"Listed {len(entries)} entries in {target}",
                    execution_time=time.time() - start_time,
                    metadata={'entries': entries},
                )

            else:
                result = EffectorResult(
                    effector_id=self.effector_id,
                    status=EffectorStatus.INVALID_COMMAND,
                    message=f"Unknown action: {command.action}",
                    execution_time=time.time() - start_time,
                )

        except Exception as e:
            result = EffectorResult(
                effector_id=self.effector_id,
                status=EffectorStatus.FAILED,
                message=str(e),
                execution_time=time.time() - start_time,
            )

        self._update_stats(result)
        return result

    def validate_command(self, command: EffectorCommand) -> Tuple[bool, str]:
        if command.action not in ('read', 'write', 'append', 'delete', 'list'):
            return False, f"Invalid action: {command.action}"

        target = command.metadata.get('target', command.parameters.get('path', ''))
        if not target:
            return False, "Missing target path"

        # Check deny patterns
        basename = os.path.basename(target)
        for pattern in self.deny_patterns:
            if fnmatch.fnmatch(basename, pattern):
                return False, f"Path matches deny pattern: {pattern}"

        # Check allowed paths (if configured)
        if self.allowed_paths:
            resolved = str(Path(target).resolve())
            if not any(resolved.startswith(str(Path(ap).resolve()))
                       for ap in self.allowed_paths):
                return False, f"Path not in allowed paths"

        return True, ""

    def is_available(self) -> bool:
        return True

    def get_info(self) -> Dict[str, Any]:
        return {
            'effector_id': self.effector_id,
            'effector_type': 'file_io',
            'supported_actions': ['read', 'write', 'append', 'delete', 'list'],
            'allowed_paths': self.allowed_paths,
            'deny_patterns': self.deny_patterns,
        }

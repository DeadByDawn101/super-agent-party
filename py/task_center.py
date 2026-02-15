# py/task_center.py
import asyncio
import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from enum import Enum
import aiofiles
import aiofiles.os
from pydantic import BaseModel

class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class SubTask(BaseModel):
    task_id: str
    parent_task_id: Optional[str] = None
    title: str
    description: str
    status: TaskStatus = TaskStatus.PENDING
    progress: int = 0  # 0-100
    result: Optional[str] = None
    error: Optional[str] = None
    created_at: str
    updated_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    agent_type: str = "default"  # 使用的智能体类型
    context: Dict[str, Any] = {}  # 额外的上下文信息

class TaskCenter:
    """任务中心 - 管理所有主任务和子任务"""
    
    def __init__(self, workspace_dir: str):
        self.workspace_dir = Path(workspace_dir)
        self.task_dir = self.workspace_dir / ".agent" / "tasks"
        self._lock = asyncio.Lock()
        self._ensure_task_dir()
    
    def _ensure_task_dir(self):
        """确保任务目录存在"""
        self.task_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_task_file(self, task_id: str) -> Path:
        """获取任务文件路径"""
        return self.task_dir / f"{task_id}.json"
    
    async def create_task(
        self,
        title: str,
        description: str,
        parent_task_id: Optional[str] = None,
        agent_type: str = "default",
        context: Optional[Dict[str, Any]] = None
    ) -> SubTask:
        """创建新任务"""
        async with self._lock:
            task_id = str(uuid.uuid4())[:8]
            now = datetime.now().isoformat()
            
            task = SubTask(
                task_id=task_id,
                parent_task_id=parent_task_id,
                title=title,
                description=description,
                created_at=now,
                updated_at=now,
                agent_type=agent_type,
                context=context or {}
            )
            
            await self._save_task(task)
            return task
    
    async def _save_task(self, task: SubTask):
        """保存任务到文件"""
        task_file = self._get_task_file(task.task_id)
        async with aiofiles.open(task_file, 'w', encoding='utf-8') as f:
            await f.write(task.model_dump_json(indent=2))
    
    async def get_task(self, task_id: str) -> Optional[SubTask]:
        """获取任务详情"""
        task_file = self._get_task_file(task_id)
        if not task_file.exists():
            return None
        
        try:
            async with aiofiles.open(task_file, 'r', encoding='utf-8') as f:
                data = await f.read()
                return SubTask.model_validate_json(data)
        except Exception as e:
            print(f"Error loading task {task_id}: {e}")
            return None
    
    async def update_task_progress(
        self,
        task_id: str,
        progress: int,
        status: Optional[TaskStatus] = None,
        result: Optional[str] = None,
        error: Optional[str] = None
    ) -> bool:
        """更新任务进度"""
        async with self._lock:
            task = await self.get_task(task_id)
            if not task:
                return False
            
            task.progress = max(0, min(100, progress))
            task.updated_at = datetime.now().isoformat()
            
            if status:
                task.status = status
                if status == TaskStatus.RUNNING and not task.started_at:
                    task.started_at = datetime.now().isoformat()
                elif status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]:
                    task.completed_at = datetime.now().isoformat()
            
            if result is not None:
                task.result = result
            
            if error is not None:
                task.error = error
                task.status = TaskStatus.FAILED
            
            await self._save_task(task)
            return True
    
    async def list_tasks(
        self,
        parent_task_id: Optional[str] = None,
        status: Optional[TaskStatus] = None
    ) -> List[SubTask]:
        """列出任务"""
        tasks = []
        
        if not self.task_dir.exists():
            return tasks
        
        for task_file in self.task_dir.glob("*.json"):
            try:
                async with aiofiles.open(task_file, 'r', encoding='utf-8') as f:
                    data = await f.read()
                    task = SubTask.model_validate_json(data)
                    
                    # 过滤条件
                    if parent_task_id is not None and task.parent_task_id != parent_task_id:
                        continue
                    if status is not None and task.status != status:
                        continue
                    
                    tasks.append(task)
            except Exception as e:
                print(f"Error loading task file {task_file}: {e}")
                continue
        
        # 按创建时间排序
        tasks.sort(key=lambda x: x.created_at, reverse=True)
        return tasks
    
    async def get_task_summary(self) -> Dict[str, Any]:
        """获取任务中心摘要"""
        all_tasks = await self.list_tasks()
        
        summary = {
            "total": len(all_tasks),
            "pending": 0,
            "running": 0,
            "completed": 0,
            "failed": 0,
            "cancelled": 0,
            "tasks": []
        }
        
        for task in all_tasks:
            summary[task.status.value] += 1
            summary["tasks"].append({
                "task_id": task.task_id,
                "title": task.title,
                "status": task.status.value,
                "progress": task.progress,
                "created_at": task.created_at,
                "updated_at": task.updated_at
            })
        
        return summary
    
    async def cancel_task(self, task_id: str) -> bool:
        """取消任务"""
        return await self.update_task_progress(
            task_id=task_id,
            progress=0,
            status=TaskStatus.CANCELLED
        )
    
    async def cleanup_old_tasks(self, days: int = 7):
        """清理旧任务（可选功能）"""
        # 实现清理逻辑...
        pass


# 全局任务中心实例字典 {workspace_path: TaskCenter}
_task_centers: Dict[str, TaskCenter] = {}

async def get_task_center(workspace_dir: str) -> TaskCenter:
    """获取或创建任务中心实例"""
    if workspace_dir not in _task_centers:
        _task_centers[workspace_dir] = TaskCenter(workspace_dir)
    return _task_centers[workspace_dir]
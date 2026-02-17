# py/task_tools.py
"""主智能体使用的任务管理工具"""
import asyncio
from typing import Optional
from py.task_center import get_task_center, TaskStatus
from py.sub_agent import run_subtask_in_background

# 1. 工具定义：创建任务
create_subtask_tool = {
    "type": "function",
    "function": {
        "name": "create_subtask",
        "description": """创建一个子任务并在后台异步执行。
⚠️ 使用场景：
- 将大任务拆分成多个独立的小任务并行执行
- 需要执行耗时较长的任务（如批量处理、深度研究）
- 结果会自动保存，无需等待

📝 返回值：子任务ID，用于后续跟踪进度""",
        "parameters": {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "子任务的简短标题"
                },
                "description": {
                    "type": "string",
                    "description": "子任务的详细描述，包含目标、背景、完成标准等"
                },
                "agent_type": {
                    "type": "string",
                    "description": "使用的智能体类型",
                    "default": "default"
                }
            },
            "required": ["title", "description"]
        }
    }
}

# 2. 工具定义：查询任务 (✅ 关键修改：添加 task_id)
query_tasks_tool = {
    "type": "function",
    "function": {
        "name": "query_task_progress",
        "description": """查询任务中心的所有任务进度，或指定查询某个特定任务的详情。

💡 使用建议：
1. 若只关心某个特定任务，请务必提供 `task_id`。
2. 若查看特定任务的详细执行过程（如代码、报错详情），请务必设置 `verbose=true`。""",
        "parameters": {
            "type": "object",
            "properties": {
                "task_id": {
                    "type": "string",
                    "description": "可选：指定唯一的任务ID。如果提供此参数，将忽略其他过滤条件，直接返回该任务的详细内容。"
                },
                "parent_task_id": {
                    "type": "string",
                    "description": "可选：指定父任务ID，只查询其子任务"
                },
                "status": {
                    "type": "string",
                    "description": "可选：过滤特定状态的任务",
                    "enum": ["pending", "running", "completed", "failed", "cancelled"]
                },
                "verbose": {
                    "type": "boolean",
                    "description": "是否显示任务的完整结果。设置为 true 可查看已完成任务的具体内容。",
                    "default": False
                }
            }
        }
    }
}

# 3. 工具定义：取消任务
cancel_subtask_tool = {
    "type": "function",
    "function": {
        "name": "cancel_subtask",
        "description": "取消一个正在执行或待执行的子任务。",
        "parameters": {
            "type": "object",
            "properties": {
                "task_id": {
                    "type": "string",
                    "description": "要取消的任务ID"
                }
            },
            "required": ["task_id"]
        }
    }
}

# --- 实现函数 ---

async def create_subtask(
    title: str,
    description: str,
    agent_type: str = "default",
    workspace_dir: str = None,
    settings: dict = None,
    parent_task_id: Optional[str] = None,
    consensus_content: Optional[str] = None
) -> str:
    """创建并启动子任务"""
    task_center = await get_task_center(workspace_dir)
    
    task = await task_center.create_task(
        title=title,
        description=description,
        parent_task_id=parent_task_id,
        agent_type=agent_type
    )
    
    asyncio.create_task(
        run_subtask_in_background(
            task_id=task.task_id,
            workspace_dir=workspace_dir,
            settings=settings, 
            consensus_content=consensus_content
        )
    )
    
    return f"✅ 子任务已创建并开始执行\n\n任务ID: {task.task_id}\n标题: {task.title}\n请不要主动查询任务进度，客户端UI会自动将当前进度和结果显示给用户"

# ✅ 关键修改：更新函数签名接收 task_id，并增加单任务查询逻辑
async def query_task_progress(
    workspace_dir: str,
    task_id: Optional[str] = None,  # 新增：接收 task_id
    parent_task_id: Optional[str] = None,
    status: Optional[str] = None,
    verbose: bool = False
) -> str:
    """查询任务进度 - 支持单任务精确查询和列表查询"""
    from py.task_center import get_task_center, TaskStatus
    
    task_center = await get_task_center(workspace_dir)
    status_enum = TaskStatus(status) if status else None
    
    tasks = []

    # 👉 分支 1：如果有 task_id，只查这一个
    if task_id:
        single_task = await task_center.get_task(task_id)
        if single_task:
            tasks = [single_task]
        else:
            return f"❌ 未找到 ID 为 {task_id} 的任务。"
    # 👉 分支 2：否则查列表
    else:
        tasks = await task_center.list_tasks(
            parent_task_id=parent_task_id,
            status=status_enum
        )
    
    if not tasks:
        return "📋 任务中心当前没有相关任务。"
    print("!!!!task:"+str(tasks))
    # 构建输出
    result_lines = [f"📋 任务中心状态 (共 {len(tasks)} 个任务)"]
    if verbose:
        result_lines.append("📢 [详情模式] 已开启：正在展示完整结果...")
    result_lines.append("-" * 30)
    
    for task in tasks:
        icon = "✅" if task.status == TaskStatus.COMPLETED else "🔄" if task.status == TaskStatus.RUNNING else "⏳"
        result_lines.append(f"{icon} [{task.task_id}] {task.title}")
        result_lines.append(f"   状态: {task.status.value.upper()} | 进度: {task.progress}%")
        
        history = task.context.get("history", [])
        # 运行中
        if task.status == TaskStatus.RUNNING:
            if history:
                result_lines.append(f"   执行动态: {history[-1]}...")
            if verbose and history:
                result_lines.append("   📜 已完成步骤:")
                for i, step in enumerate(history, 1):
                    result_lines.append(f"     {i}. {step}...")

        # 已完成
        elif task.status == TaskStatus.COMPLETED:
            if verbose:
                # ✅ 如果 verbose=True，强制显示完整 result
                result_content = task.result if task.result else "（无结果内容）"
                result_lines.append(f"   🎯 最终完整产出:\n{result_content}\n")
                
                # 可选：显示中间过程
                if history:
                    result_lines.append("   📜 执行过程回溯 (最近3步):")
                    for i, step in enumerate(history[-3:], 1):
                        result_lines.append(f"     ... {step} ...")
            else:
                summary = task.context.get('summary') or (task.result if task.result else "无结果内容")
                result_lines.append(f"   📝 结果摘要: {summary}")
                result_lines.append(f"   💡 (提示: 使用 verbose=true 可查看完整报告)")

        elif task.status == TaskStatus.FAILED:
            result_lines.append(f"   ❌ 错误信息: {task.error}")

        result_lines.append("") 

    return "\n".join(result_lines)

async def cancel_subtask(workspace_dir: str, task_id: str) -> str:
    """取消子任务"""
    task_center = await get_task_center(workspace_dir)
    success = await task_center.cancel_task(task_id)
    return f"✅ 任务 {task_id} 已取消" if success else f"❌ 取消任务 {task_id} 失败"
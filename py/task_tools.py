# py/task_tools.py
"""主智能体使用的任务管理工具"""
import asyncio
from typing import Optional
from py.task_center import get_task_center, TaskStatus
from py.sub_agent import run_subtask_in_background

create_subtask_tool = {
    "type": "function",
    "function": {
        "name": "create_subtask",
        "description": """创建一个子任务并在后台异步执行。

⚠️ 使用场景：
- 将大任务拆分成多个独立的小任务并行执行
- 需要执行耗时较长的任务（如批量处理、深度研究）
- 需要委托给专门的子智能体处理特定领域问题

✅ 特点：
- 异步执行，不阻塞主对话
- 自动保存进度，重启后可恢复
- 可通过 query_task_progress 查看实时状态

📝 返回值：子任务ID，用于后续跟踪进度

⚠️ 注意：
- 每个子任务都是独立的对话上下文
- 子任务无法访问主对话的历史记录（除非在description中明确说明）
- 建议在description中提供完整的背景信息和明确的完成标准
- 如果不是用户要求，子任务创建后你无需主动查询其进度，客户端UI会自动将当前进度和结果显示给用户""",
        "parameters": {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "子任务的简短标题（建议不超过50字）"
                },
                "description": {
                    "type": "string",
                    "description": """子任务的详细描述，必须包含：
1. 任务目标：期望达成的具体结果
2. 背景信息：必要的上下文和前置知识
3. 完成标准：如何判断任务已完成
4. 约束条件：需要遵守的规则或限制

示例：
\"\"\"
任务目标：分析 data.csv 文件中的销售数据，生成月度报告

背景信息：
- 文件位于 ./reports/data.csv
- 包含列：date, product, quantity, revenue
- 需要关注 2024年1月-3月的数据

完成标准：
- 生成包含趋势图的 Markdown 报告
- 计算出每月总销售额和增长率
- 识别销售额 Top 3 产品

约束条件：
- 使用 Python pandas 进行数据处理
- 图表使用 matplotlib 生成
- 报告保存为 monthly_report.md
\"\"\""""
                },
                "agent_type": {
                    "type": "string",
                    "description": "使用的智能体类型（当前固定为 'default'，未来可扩展）",
                    "default": "default"
                }
            },
            "required": ["title", "description"]
        }
    }
}

query_tasks_tool = {
    "type": "function",
    "function": {
        "name": "query_task_progress",
        "description": """查询任务中心的所有任务进度和状态。

返回信息：
- ⏳ pending: 等待执行
- 🔄 running: 正在执行中
- ✅ completed: 已完成（可查看结果）
- ❌ failed: 执行失败（可查看错误信息）
- 🚫 cancelled: 已取消

每个任务包含：
- 任务ID、标题、状态
- 进度百分比（0-100%）
- 创建时间、最后更新时间
- 执行结果或错误信息（如有）

💡 使用建议：
- 创建子任务后，定期查询进度
- 在回复用户前，先确认所有相关子任务已完成
- 如果子任务失败，分析错误并决定是否重试或调整策略""",
        "parameters": {
            "type": "object",
            "properties": {
                "parent_task_id": {
                    "type": "string",
                    "description": "可选：指定父任务ID，只查询其子任务"
                },
                "status": {
                    "type": "string",
                    "description": "可选：过滤特定状态的任务",
                    "enum": ["pending", "running", "completed", "failed", "cancelled"]
                }
            }
        }
    }
}

cancel_subtask_tool = {
    "type": "function",
    "function": {
        "name": "cancel_subtask",
        "description": """取消一个正在执行或待执行的子任务。

⚠️ 注意：
- 只能取消 pending 或 running 状态的任务
- 已完成(completed)或已失败(failed)的任务无法取消
- 取消操作是异步的，可能需要几秒钟生效

💡 使用场景：
- 发现任务定义有误，需要重新创建
- 任务执行时间过长，需要中止
- 用户改变需求，不再需要该任务的结果""",
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

# 工具实现函数

async def create_subtask(
    title: str,
    description: str,
    agent_type: str = "default",
    workspace_dir: str = None,
    settings: dict = None,
    parent_task_id: Optional[str] = None,
    consensus_content: Optional[str] = None
) -> str:
    """创建并启动子任务（不需要 base_url）"""
    task_center = await get_task_center(workspace_dir)
    
    # 创建任务
    task = await task_center.create_task(
        title=title,
        description=description,
        parent_task_id=parent_task_id,
        agent_type=agent_type
    )
    
    # 在后台异步执行（SubAgentExecutor 内部会自动获取端口）
    asyncio.create_task(
        run_subtask_in_background(
            task_id=task.task_id,
            workspace_dir=workspace_dir,
            settings=settings, 
            consensus_content=consensus_content
        )
    )
    
    return f"✅ 子任务已创建并开始执行\n\n任务ID: {task.task_id}\n标题: {task.title}\n\n可以使用 query_task_progress 工具查询进度。"

async def query_task_progress(
    workspace_dir: str,
    parent_task_id: Optional[str] = None,
    status: Optional[str] = None
) -> str:
    """⭐ 查询任务进度 - 会话隔离版（只显示摘要）"""
    task_center = await get_task_center(workspace_dir)
    
    status_enum = TaskStatus(status) if status else None
    
    tasks = await task_center.list_tasks(
        parent_task_id=parent_task_id,
        status=status_enum
    )
    
    if not tasks:
        return "📋 当前没有任务"
    
    result = f"📋 任务中心状态 (共 {len(tasks)} 个任务)\n\n"
    
    status_icons = {
        "pending": "⏳",
        "running": "🔄",
        "completed": "✅",
        "failed": "❌",
        "cancelled": "🚫"
    }
    
    for task in tasks:
        icon = status_icons.get(task.status.value, "❓")
        result += f"{icon} [{task.task_id}] {task.title}\n"
        result += f"   状态: {task.status.value} | 进度: {task.progress}%\n"
        
        # ⭐ 关键改进：完成的任务只显示摘要
        if task.status == TaskStatus.COMPLETED:
            # 尝试从 context 中获取摘要
            summary = task.context.get('summary')
            
            if summary:
                result += f"   📝 摘要: {summary}\n"
            else:
                # 降级：显示 result 的前 150 字
                if task.result:
                    short_result = task.result[:150]
                    if len(task.result) > 150:
                        short_result += "..."
                    result += f"   📝 结果: {short_result}\n"
            
            # 提示：完整结果在哪里
            result += f"   💡 完整结果保存在: .agent/tasks/{task.task_id}.json\n"
        
        if task.status == TaskStatus.FAILED and task.error:
            result += f"   ❌ 错误: {task.error}\n"
        
        result += f"   创建: {task.created_at.split('T')[0]} {task.created_at.split('T')[1][:8]}\n"
        result += f"   更新: {task.updated_at.split('T')[0]} {task.updated_at.split('T')[1][:8]}\n"
        result += "\n"
    
    return result

async def cancel_subtask(workspace_dir: str, task_id: str) -> str:
    """取消子任务"""
    task_center = await get_task_center(workspace_dir)
    
    success = await task_center.cancel_task(task_id)
    
    if success:
        return f"✅ 任务 {task_id} 已取消"
    else:
        return f"❌ 取消任务 {task_id} 失败（任务不存在或已完成）"
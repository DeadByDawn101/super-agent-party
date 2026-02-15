# py/sub_agent.py - 修正版

import asyncio
import json
import httpx
from typing import Dict, List, Optional, Any
from py.task_center import get_task_center, TaskStatus
from py.get_setting import load_settings, get_port

class SubAgentExecutor:
    """子智能体执行器"""
    
    def __init__(self, workspace_dir: str, settings: Dict):
        self.workspace_dir = workspace_dir
        self.settings = settings
        self.port = get_port()
        self.base_url = f"http://127.0.0.1:{self.port}"
        self.chat_endpoint = f"{self.base_url}/v1/chat/completions"
        self.simple_chat_endpoint = f"{self.base_url}/simple_chat"
    
    async def execute_subtask(
        self,
        task_id: str,
        consensus_content: Optional[str] = None,
        max_iterations: int = 15
    ) -> Dict[str, Any]:
        """执行子任务的主循环"""
        task_center = await get_task_center(self.workspace_dir)
        task = await task_center.get_task(task_id)
        
        if not task:
            return {"success": False, "error": f"Task {task_id} not found"}
        
        # 标记任务开始
        await task_center.update_task_progress(
            task_id=task_id,
            progress=0,
            status=TaskStatus.RUNNING
        )
        
        iteration = 0
        conversation_history = []
        
        # 构建初始系统提示
        system_prompt = self._build_system_prompt(task, consensus_content)
        conversation_history.append({
            "role": "system",
            "content": system_prompt
        })
        
        # 初始用户消息
        initial_message = f"""请执行以下任务：

{task.description}

要求：
1. 专注完成任务目标
2. 使用所有可用工具
3. 完成后直接给出结果
"""
        conversation_history.append({
            "role": "user",
            "content": initial_message
        })
        
        try:
            # 增加超时时间，防止长思考/工具执行导致断连
            async with httpx.AsyncClient(timeout=600.0) as http_client:
                while iteration < max_iterations:
                    iteration += 1
                    current_progress = int((iteration / max_iterations) * 70)
                    
                    await task_center.update_task_progress(
                        task_id=task_id,
                        progress=current_progress,
                        status=TaskStatus.RUNNING
                    )
                    
                    print(f"[SubAgent] Task {task_id} - Iteration {iteration} (Streaming)")
                    
                    # ---------------------------------------------------------
                    # ⭐ 核心修改：只负责发消息和收文本，不做任何工具处理
                    # ---------------------------------------------------------
                    assistant_response = await self._call_llm_stream_only(
                        http_client=http_client,
                        messages=conversation_history,
                        model='super-model' # 你的后端逻辑会接管这个请求
                    )
                    
                    # 记录助手回复
                    conversation_history.append({
                        "role": "assistant",
                        "content": assistant_response
                    })
                    
                    # ---------------------------------------------------------
                    # 智能判断任务是否完成（非流式，快速判断）
                    # ---------------------------------------------------------
                    is_complete = await self._check_task_completion_smart(
                        task=task,
                        conversation_history=conversation_history,
                        http_client=http_client
                    )
                    
                    if is_complete:
                        print(f"[SubAgent] Task {task_id} - Completed")
                        
                        await task_center.update_task_progress(
                            task_id=task_id,
                            progress=90,
                            status=TaskStatus.RUNNING
                        )
                        
                        # 提取结果和摘要
                        result_dict = await self._extract_final_result(
                            task=task,
                            conversation_history=conversation_history,
                            http_client=http_client
                        )

                        task.context['summary'] = result_dict['summary']
                        
                        await task_center.update_task_progress(
                            task_id=task_id,
                            progress=100,
                            status=TaskStatus.COMPLETED,
                            result=result_dict['full']
                        )

                        return {
                            "success": True,
                            "task_id": task_id,
                            "result": result_dict['full'],
                            "summary": result_dict['summary'],
                            "iterations": iteration
                        }

                    # 未完成，继续催促（这会让后端 Agent 继续下一轮思考）
                    conversation_history.append({
                        "role": "user",
                        "content": "请继续执行下一步。"
                    })
                
                # 超时处理
                await task_center.update_task_progress(
                    task_id=task_id,
                    progress=100,
                    status=TaskStatus.FAILED,
                    error=f"超过最大迭代次数({max_iterations})"
                )
                return {"success": False, "error": "Max iterations reached"}

        except Exception as e:
            error_msg = f"执行出错: {str(e)}"
            print(f"[SubAgent] Critical Error: {error_msg}")
            await task_center.update_task_progress(
                task_id=task_id,
                progress=0,
                status=TaskStatus.FAILED,
                error=error_msg
            )
            return {"success": False, "error": error_msg}

    async def _call_llm_stream_only(
        self, 
        http_client: httpx.AsyncClient, 
        messages: List[Dict], 
        model: str
    ) -> str:
        """
        ⭐ 健壮的流式接收器
        只收集 'content' 文本，忽略后端发送的工具状态、思考过程等中间数据
        解决 KeyError: 'choices' 问题
        """
        payload = {
            "messages": messages,
            "model": model,
            "stream": True, 
            "temperature": 0.5,
            "max_tokens": self.settings.get('max_tokens', 4000),
            "is_sub_agent": True,
            # 子智能体不需要自己能调用子任务，防止递归
            "disable_tools": ["create_subtask", "query_tasks_tool", "cancel_subtask"] 
        }

        full_content = ""

        try:
            async with http_client.stream("POST", self.chat_endpoint, json=payload, headers={"Content-Type": "application/json"}) as response:
                
                if response.status_code != 200:
                    error_text = await response.aread()
                    raise Exception(f"API Error {response.status_code}: {error_text.decode('utf-8')}")

                async for line in response.aiter_lines():
                    if not line.strip():
                        continue
                    if line.startswith("data: "):
                        data_str = line[6:]
                        if data_str.strip() == "[DONE]":
                            break
                        
                        try:
                            chunk = json.loads(data_str)
                            
                            # 1. 检查错误
                            if "error" in chunk:
                                # 如果是流中间的错误，打印但尝试继续（或者抛出）
                                print(f"[SubAgent Stream Error] {chunk['error']}")
                                continue

                            # 2. 安全检查 choices
                            # 后端发来的 tool_content, tool_progress 也会包含 choices，但 delta 结构不同
                            if not chunk.get("choices"):
                                continue
                            
                            choice = chunk["choices"][0]
                            delta = choice.get("delta", {})

                            # 3. 这里的关键是：只提取 content
                            # generate_stream_response 在执行工具时会发送:
                            # {"tool_content": ...} -> 忽略
                            # {"tool_progress": ...} -> 忽略
                            # {"tool_calls": ...} -> 忽略 (后端自己会处理，我们只等结果)
                            
                            content = delta.get("content")
                            if content:
                                full_content += content
                                
                            # 可选：如果你想在控制台看到子智能体在干嘛，可以打印 tool_content
                            # if delta.get("tool_content"):
                            #     print(f"  [SubAgent Tool] {delta['tool_content'].get('title')}")

                        except json.JSONDecodeError:
                            continue
                        except Exception:
                            # 忽略单行解析错误，防止整个任务崩掉
                            continue

        except Exception as e:
            raise Exception(f"Stream Failed: {str(e)}")

        if not full_content:
            # 如果跑了一圈没有文本（比如纯工具执行完了但还没说话），给个占位符
            # 但通常 Agent 最后都会总结发言
            return "(任务执行中，无文本输出)"

        return full_content

    # ---------------- 以下辅助方法保持不变 ----------------
    
    def _build_system_prompt(self, task, consensus_content: Optional[str]) -> str:
        prompt = f"""你是一个专业的任务执行助手。
【任务信息】ID: {task.task_id} | 标题: {task.title}
【执行要求】专注完成任务，使用可用工具，完成后明确表示结束。"""
        if consensus_content:
            prompt += f"\n\n【共识规范】\n{consensus_content}\n"
        return prompt
    
    async def _check_task_completion_smart(self, task, conversation_history, http_client) -> bool:
        # 使用 simple_chat (非流式) 进行快速判断
        recent = self._get_recent_conversation(conversation_history)
        msgs = [
            {"role": "system", "content": "判断任务是否完成，只回复YES或NO。"},
            {"role": "user", "content": f"任务：{task.description}\n最近进展：{recent}\n是否完成？"}
        ]
        try:
            resp = await http_client.post(self.simple_chat_endpoint, json={"messages": msgs, "model": "super-model", "stream": False})
            if resp.status_code == 200:
                data = resp.json()
                if "choices" in data:
                    return data["choices"][0]["message"]["content"].strip().upper().startswith("YES")
        except: pass
        return False
    
    async def _extract_final_result(self, task, conversation_history, http_client) -> Dict[str, str]:
        # 简化版提取逻辑
        full_res = "提取失败"
        try:
            msgs = [{"role": "user", "content": f"基于历史对话提取任务【{task.description}】的最终结果："}]
            # 这里需要把 history 塞进去，省略代码...
            # 为简单起见，直接取最后一条 assistant 消息
            for msg in reversed(conversation_history):
                if msg["role"] == "assistant":
                    full_res = msg["content"]
                    break
        except: pass
        
        summary = full_res[:100] + "..."
        return {"full": full_res, "summary": summary}

    def _get_recent_conversation(self, conversation_history: List[Dict]) -> str:
        texts = []
        for msg in reversed(conversation_history[-5:]):
            texts.append(f"{msg['role']}: {str(msg.get('content'))[:200]}")
        return "\n".join(texts)

# 保持 run_subtask_in_background
async def run_subtask_in_background(task_id: str, workspace_dir: str, settings: Dict, consensus_content: Optional[str] = None):
    executor = SubAgentExecutor(workspace_dir, settings)
    await executor.execute_subtask(task_id, consensus_content)
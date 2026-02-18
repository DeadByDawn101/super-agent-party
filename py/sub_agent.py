# py/sub_agent.py

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
        """执行子任务的主循环 - 增强实时进度反馈版"""
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
        # 初始化展示历史，从 context 恢复（如果已有）
        assistant_only_history = task.context.get("history", [])
        
        # 1. 构建初始上下文
        system_prompt = self._build_system_prompt(task, consensus_content)
        conversation_history.append({"role": "system", "content": system_prompt})
        
        initial_user_msg = f"请执行以下任务：\n\n{task.description}\n\n要求：完成后请整理出最终结果。"
        conversation_history.append({"role": "user", "content": initial_user_msg})
        
        try:
            async with httpx.AsyncClient(timeout=600.0) as http_client:
                while iteration < max_iterations:
                    iteration += 1
                    # 计算大轮次的基础进度
                    current_progress = 10 + int((iteration / max_iterations) * 80)
                    
                    print(f"[SubAgent] Task {task_id} - Iteration {iteration}")
                    
                    # 2. 调用 LLM 获取助手回复 (内部会实时更新进度和 history)
                    assistant_response = await self._call_llm_stream_only(
                        http_client=http_client,
                        messages=conversation_history,
                        model='super-model',
                        task_id=task_id,            # 传入用于实时更新
                        task_center=task_center,    # 传入用于实时更新
                        base_progress=current_progress,
                        display_history=assistant_only_history
                    )
                    
                    # 3. 记录到内存完整历史中
                    conversation_history.append({
                        "role": "assistant",
                        "content": assistant_response
                    })
                    
                    # 4. 再次同步最终历史（确保本轮最后一段文本被记入）
                    await task_center.update_task_progress(
                        task_id=task_id,
                        progress=current_progress,
                        status=TaskStatus.RUNNING,
                        context={
                            "history": assistant_only_history,
                            "current_iteration": iteration
                        }
                    )
                    
                    # 5. 智能判断任务是否完成
                    is_complete = await self._check_task_completion_smart(
                        task=task,
                        conversation_history=conversation_history,
                        http_client=http_client
                    )
                    
                    if is_complete:
                        print(f"[SubAgent] Task {task_id} - Completed internally. Extracting final result...")
                        
                        # 6. 提取最终结果和摘要
                        result_dict = await self._extract_final_result(
                            task=task,
                            conversation_history=conversation_history,
                            http_client=http_client
                        )

                        await task_center.update_task_progress(
                            task_id=task_id,
                            progress=100,
                            status=TaskStatus.COMPLETED,
                            result=result_dict['full'],
                            context={
                                "summary": result_dict['summary'],
                                "history": assistant_only_history
                            }
                        )

                        return {
                            "success": True,
                            "task_id": task_id,
                            "result": result_dict['full'],
                            "summary": result_dict['summary'],
                            "iterations": iteration
                        }

                    # 未完成，添加用户指令引导下一轮
                    conversation_history.append({
                        "role": "user",
                        "content": "请继续执行任务。如果已完成所有步骤，请总结并给出最终结果。"
                    })
                
                error_msg = f"超过最大迭代次数({max_iterations})，任务强制结束。"
                await task_center.update_task_progress(
                    task_id=task_id,
                    progress=100,
                    status=TaskStatus.FAILED,
                    error=error_msg
                )
                return {"success": False, "error": error_msg}

        except Exception as e:
            error_msg = f"执行过程中发生异常: {str(e)}"
            print(f"[SubAgent] Error: {error_msg}")
            await task_center.update_task_progress(task_id=task_id, progress=0, status=TaskStatus.FAILED, error=error_msg)
            return {"success": False, "error": error_msg}

    async def _call_llm_stream_only(
        self, 
        http_client: httpx.AsyncClient, 
        messages: List[Dict], 
        model: str,
        task_id: str = None,
        task_center: Any = None,
        base_progress: int = 0,
        display_history: List[str] = None
    ) -> str:
        """
        ⭐ 健壮的流式接收器 - 实时更新版
        在解析流的过程中，捕获工具执行状态并实时写入 TaskCenter，
        解决了子任务在 main.py 内部循环时界面不更新的问题。
        """
        payload = {
            "messages": messages,
            "model": model,
            "stream": True, 
            "temperature": 0.5,
            "max_tokens": self.settings.get('max_tokens', 4000),
            "is_sub_agent": True,
            "disable_tools": ["create_subtask", "query_tasks_tool", "cancel_subtask"] 
        }

        full_content = ""
        current_text_buffer = "" # 用于缓冲连续的助手文本
        tool_step_counter = 0

        try:
            async with http_client.stream("POST", self.chat_endpoint, json=payload, headers={"Content-Type": "application/json"}) as response:
                if response.status_code != 200:
                    error_text = await response.aread()
                    raise Exception(f"API Error {response.status_code}: {error_text.decode('utf-8')}")

                async for line in response.aiter_lines():
                    if not line.strip(): continue
                    if line.startswith("data: "):
                        data_str = line[6:]
                        if data_str.strip() == "[DONE]": break
                        try:
                            chunk = json.loads(data_str)
                            if "error" in chunk: continue
                            if not chunk.get("choices"): continue
                            
                            choice = chunk["choices"][0]
                            delta = choice.get("delta", {})

                            # 1. 累积助手回复的文本内容
                            content = delta.get("content")
                            if content:
                                full_content += content
                                current_text_buffer += content

                            # 2. ⭐ 实时捕获工具执行详情
                            tool_data = delta.get("tool_content")
                            if tool_data and task_center and task_id:
                                # 只要触发了工具，说明之前的文本思考告一段落，存入 history
                                if current_text_buffer.strip():
                                    display_history.append(current_text_buffer.strip())
                                    current_text_buffer = "" # 清空缓冲
                                
                                tool_type = tool_data.get("type")
                                tool_title = tool_data.get("title", "Unknown Tool")
                                
                                # 只处理“结果”类信息
                                if tool_type in ["tool_result", "error"]:
                                    tool_step_counter += 1
                                    res_content = tool_data.get("content", "")
                                    icon = "✅" if tool_type == "tool_result" else "❌"
                                    
                                    # 记录到 history (截断过长结果)
                                    short_res = str(res_content)[:300] + "..." if len(str(res_content)) > 300 else str(res_content)
                                    display_history.append(f"{icon} [{tool_title}]\nResult: {short_res}")
                                    
                                    # 微调进度：每完成一个工具增加 2%
                                    micro_progress = min(base_progress + (tool_step_counter * 2), 99)
                                    
                                    # ⭐ 核心：在此处直接实时更新 JSON 文件
                                    await task_center.update_task_progress(
                                        task_id=task_id,
                                        progress=micro_progress,
                                        status=TaskStatus.RUNNING,
                                        context={"history": display_history}
                                    )

                        except: continue

        except Exception as e:
            raise Exception(f"Stream Failed: {str(e)}")

        # 流结束，如果缓冲区还有文本，补录进去
        if current_text_buffer.strip() and display_history is not None:
            display_history.append(current_text_buffer.strip())

        return full_content if full_content else "(任务执行中...)"

    # ---------------- 辅助方法 ----------------
    
    def _build_system_prompt(self, task, consensus_content: Optional[str]) -> str:
        prompt = f"你是一个专业的任务执行助手。\n【任务信息】ID: {task.task_id} | 标题: {task.title}\n【执行要求】专注完成任务，使用可用工具，完成后明确表示结束。"
        if consensus_content: prompt += f"\n\n【共识规范】\n{consensus_content}\n"
        return prompt
    
    async def _check_task_completion_smart(self, task, conversation_history, http_client) -> bool:
        recent = self._get_recent_conversation(conversation_history)
        msgs = [{"role": "system", "content": "判断任务是否完成，只回复YES或NO。"},
                {"role": "user", "content": f"任务：{task.description}\n最近进展：{recent}\n是否完成？"}]
        try:
            resp = await http_client.post(self.simple_chat_endpoint, json={"messages": msgs, "model": "super-model", "stream": False})
            if resp.status_code == 200:
                data = resp.json()
                return data["choices"][0]["message"]["content"].strip().upper().startswith("YES")
        except: pass
        return False
    
    async def _extract_final_result(self, task, conversation_history, http_client) -> Dict[str, str]:
        history_str = ""
        for msg in conversation_history:
            if msg["role"] in ["assistant", "user"]:
                content = msg["content"] if msg["content"] else "[执行了工具操作]"
                history_str += f"{msg['role']}: {content}\n"
        msgs = [{"role": "system", "content": "请从对话历史中提取出任务的【最终执行结果】，保留核心干货（如报告内容、代码、分析结果）。"},
                {"role": "user", "content": f"任务目标：{task.description}\n\n对话历史：\n{history_str[-6000:]}\n\n请给出最终结果："}]
        full_res = "未提取到结果"
        try:
            resp = await http_client.post(self.simple_chat_endpoint, json={"messages": msgs, "model": "super-model", "stream": False})
            if resp.status_code == 200: full_res = resp.json()["choices"][0]["message"]["content"].strip()
        except: 
            full_res = "\n".join([m["content"] for m in conversation_history if m["role"] == "assistant" and m["content"]])
        return {"full": full_res, "summary": full_res[:200].replace("\n", " ") + "..."}

    def _get_recent_conversation(self, conversation_history: List[Dict]) -> str:
        texts = []
        for msg in reversed(conversation_history[-5:]):
            texts.append(f"{msg['role']}: {str(msg.get('content'))[:200]}")
        return "\n".join(texts)

async def run_subtask_in_background(task_id: str, workspace_dir: str, settings: Dict, consensus_content: Optional[str] = None):
    executor = SubAgentExecutor(workspace_dir, settings)
    await executor.execute_subtask(task_id, consensus_content)
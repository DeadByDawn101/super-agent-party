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
        """执行子任务的主循环 - 增强版"""
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
        
        # 1. 构建初始上下文
        system_prompt = self._build_system_prompt(task, consensus_content)
        conversation_history.append({"role": "system", "content": system_prompt})
        
        initial_user_msg = f"请执行以下任务：\n\n{task.description}\n\n要求：完成后请整理出最终结果。"
        conversation_history.append({"role": "user", "content": initial_user_msg})
        
        try:
            async with httpx.AsyncClient(timeout=600.0) as http_client:
                while iteration < max_iterations:
                    iteration += 1
                    # 计算展示进度 (10% - 90%)
                    current_progress = 10 + int((iteration / max_iterations) * 80)
                    
                    print(f"[SubAgent] Task {task_id} - Iteration {iteration}")
                    
                    # 2. 调用 LLM 获取助手回复 (流式接收文本)
                    assistant_response = await self._call_llm_stream_only(
                        http_client=http_client,
                        messages=conversation_history,
                        model='super-model'
                    )
                    
                    # 3. 记录到内存完整历史中（用于给模型维持上下文）
                    conversation_history.append({
                        "role": "assistant",
                        "content": assistant_response
                    })
                    
                    # 4. ⭐ 提取助手内容并同步到 JSON 的 context['history']
                    # 只提取 assistant 的消息内容
                    assistant_only_history = [
                        m["content"] for m in conversation_history 
                        if m["role"] == "assistant" and m["content"]
                    ]
                    
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
                        
                        # 6. ⭐ 提取最终结果和摘要
                        result_dict = await self._extract_final_result(
                            task=task,
                            conversation_history=conversation_history,
                            http_client=http_client
                        )

                        # 最终更新：填入 result，保存 summary 和 完整的助手历史
                        await task_center.update_task_progress(
                            task_id=task_id,
                            progress=100,
                            status=TaskStatus.COMPLETED,
                            result=result_dict['full'], # 只有这里会写入最终结果
                            context={
                                "summary": result_dict['summary'],
                                "history": assistant_only_history # 最终的历史记录
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
                
                # 循环结束（达到最大次数）
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
        """使用 AI 总结整个对话作为最终结果，确保不遗漏关键信息"""
        
        # 构建一个专门的 Prompt 让 AI 整理结果
        history_str = ""
        for msg in conversation_history:
            if msg["role"] in ["assistant", "user"]:
                content = msg["content"] if msg["content"] else "[执行了工具操作]"
                history_str += f"{msg['role']}: {content}\n"

        msgs = [
            {"role": "system", "content": "你是一个结果提取专家。请从对话历史中提取出任务的【最终执行结果】。去除所有过程描述和“任务已完成”之类的废话，保留核心干货（如报告内容、代码、分析结果）。"},
            {"role": "user", "content": f"任务目标：{task.description}\n\n对话历史：\n{history_str[-6000:]}\n\n请给出最终结果："}
        ]
        
        full_res = "未提取到结果"
        try:
            resp = await http_client.post(
                self.simple_chat_endpoint, 
                json={"messages": msgs, "model": "super-model", "stream": False}
            )
            if resp.status_code == 200:
                data = resp.json()
                full_res = data["choices"][0]["message"]["content"].strip()
        except Exception as e:
            print(f"提取失败: {e}")
            # 降级方案：合并所有助手的文本
            full_res = "\n".join([m["content"] for m in conversation_history if m["role"] == "assistant" and m["content"]])

        # 生成一个更智能的摘要
        summary = full_res[:200].replace("\n", " ") + "..."
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
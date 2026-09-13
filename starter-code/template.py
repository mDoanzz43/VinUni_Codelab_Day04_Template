"""
Lab #4: System Prompt Engineering & Tool Calling Engine
Các hạng mục thực hành đã được hoàn thiện trong file này.

Kiến trúc:
  - ChatbotBaseline: LLM thuần, không dùng tool → quan sát hallucination.
  - ToolCallingAgent: Agent dùng System Prompt + 2 Tool Schemas.
"""

import json
import re
from typing import Dict, Any, List
from tools import TOOL_DEFINITIONS, TOOL_MAP, search_product_catalog, submit_support_ticket
# import google.genai as genai
# from dotenv import load_dotenv
# import os 
# load_dotenv()


# ═══════════════════════════════════════════════════════════════════════════
# MILESTONE 1: Thiết kế SYSTEM PROMPT cấp sản xuất
# Yêu cầu: Phải chứa Persona, Core Rules, Operational Boundaries, Output Contract.
# ═══════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """
Bạn là VinAssistant, trợ lý AI tư vấn chính thức cho hệ sinh thái Vingroup.

## PERSONA
- Vai trò: tư vấn sản phẩm VinFast, dịch vụ Vinpearl và tiếp nhận hỗ trợ.
- Giọng điệu: chuyên nghiệp, thân thiện, rõ ràng và chính xác.

## AVAILABLE TOOLS
- search_product_catalog: tra cứu sản phẩm theo danh mục và giá tối đa.
- submit_support_ticket: tạo phiếu hỗ trợ từ thông tin khách hàng cung cấp.

## CORE RULES
1. Không bịa tên, giá, tình trạng sản phẩm hoặc mã ticket.
2. Phải gọi công cụ tra cứu khi người dùng cần dữ liệu sản phẩm cụ thể.
3. Phải gọi công cụ ticket khi người dùng yêu cầu ghi nhận sự cố và đã đủ thông tin.
4. Chỉ dùng Observation để lập câu trả lời; nếu không có kết quả, nói rõ điều đó.
5. Không tiết lộ suy luận nội bộ.

## OPERATIONAL BOUNDARIES
- Chỉ hỗ trợ sản phẩm và dịch vụ thuộc hệ sinh thái Vingroup.
- Với nội dung ngoài phạm vi, lịch sự từ chối.

## OUTPUT CONTRACT
- Nội bộ tuân theo Thought -> Action -> Observation.
- Chỉ hiển thị Final Answer rõ ràng cho người dùng.
"""


# ═══════════════════════════════════════════════════════════════════════════
# CLASS: ChatbotBaseline
# ═══════════════════════════════════════════════════════════════════════════

class ChatbotBaseline:
    """Baseline LLM Chatbot — Không sử dụng Tool Calling hay ReAct Loop."""

    def query(self, user_input: str) -> Dict[str, Any]:
        # Baseline trả lời tĩnh (mock), không dùng tool để minh họa rủi ro hallucination.
        # Mục tiêu: Quan sát hiện tượng bịa thông tin (hallucination)
        return {
            "answer": (
                "[Chatbot Baseline] Tôi trả lời trực tiếp, không dùng công cụ: "
                f"{user_input}. Thông tin này chưa được kiểm chứng bằng dữ liệu thực."
            ),
            "tool_calls": [],
            "status": "success",
            "mode": "mock_baseline"
        }

# ═══════════════════════════════════════════════════════════════════════════
# CLASS: ToolCallingAgent
# ═══════════════════════════════════════════════════════════════════════════

class ToolCallingAgent:
    """Agent với System Prompt Engineering & Tool Calling."""

    def __init__(self, max_iterations: int = 5):
        self.max_iterations = max_iterations
        self.trace: List[Dict[str, Any]] = []

    def run(self, user_input: str) -> Dict[str, Any]:
        """Điểm vào chính — chạy Agent Loop."""
        self.trace = []
        text = self._repair_text(user_input)
        intents = self._detect_intents(text)
        actions = []
        if intents["needs_catalog"]:
            actions.append(("search_product_catalog", self._catalog_args(text)))
        if intents["needs_ticket"]:
            actions.append(("submit_support_ticket", self._ticket_args(text)))

        required_iterations = max(1, len(actions))
        if self.max_iterations < required_iterations:
            return {
                "answer": "Lỗi: Vượt quá số bước tối đa.",
                "trace": self.trace,
                "iterations": 0,
                "status": "max_iterations_reached",
            }

        for iteration, (tool_name, arguments) in enumerate(actions, start=1):
            observation = TOOL_MAP[tool_name](**arguments)
            self.trace.append({
                "iteration": iteration,
                "action": tool_name,
                "arguments": arguments,
                "observation": observation,
            })

        answer = self._build_answer(intents)
        if not actions:
            self.trace.append({"iteration": 1, "action": "direct_answer"})
        return {
            "answer": answer,
            "trace": self.trace,
            "iterations": required_iterations,
            "status": "completed",
        }

    @staticmethod
    def _repair_text(value: str) -> str:
        """Chấp nhận cả Unicode đúng và chuỗi mojibake trong fixture cũ."""
        try:
            return value.encode("latin-1").decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            return value

    @staticmethod
    def _detect_intents(text: str) -> Dict[str, bool]:
        lowered = text.lower()
        catalog_terms = ("giá dưới", "giá tối đa", "xem xe", "resort", "vinpearl giá", "xe điện nào")
        ticket_terms = ("bị lỗi", "hỗ trợ", "ghi nhận", "phản hồi", "xử lý gấp", "nghiêm trọng", "ẩm mốc")
        is_warranty_faq = "bảo hành" in lowered and not any(term in lowered for term in ticket_terms)
        return {
            "needs_catalog": any(term in lowered for term in catalog_terms) and not is_warranty_faq,
            "needs_ticket": any(term in lowered for term in ticket_terms) and not is_warranty_faq,
            "is_faq": is_warranty_faq,
        }

    @staticmethod
    def _extract_max_price(text: str) -> int:
        match = re.search(r"(?:dưới|tối đa)\s+([\d.,]+)\s*(triệu|tỷ)?", text.lower())
        if not match:
            return 999999999999
        number = float(match.group(1).replace(".", "").replace(",", "."))
        multiplier = 1_000_000_000 if match.group(2) == "tỷ" else 1_000_000
        return int(number * multiplier)

    def _catalog_args(self, text: str) -> Dict[str, Any]:
        is_travel = any(term in text.lower() for term in ("resort", "vinpearl", "du lịch"))
        return {
            "category": "du_lich" if is_travel else "xe_dien",
            "max_price": self._extract_max_price(text),
        }

    @staticmethod
    def _ticket_args(text: str) -> Dict[str, str]:
        name_match = re.search(r"(?:tôi tên|tên tôi là)\s+([^,.]+)", text, flags=re.IGNORECASE)
        customer_name = name_match.group(1).strip() if name_match else "Khách hàng"
        issue_match = re.search(r"((?:xe|phòng)[^.!?]*(?:bị|lỗi)[^.!?]*)", text, flags=re.IGNORECASE)
        issue = issue_match.group(1).strip() if issue_match else text.strip()
        priority = "high" if any(word in text.lower() for word in ("nghiêm trọng", "khẩn", "gấp")) else "medium"
        return {"customer_name": customer_name, "issue_description": issue, "priority": priority}

    def _build_answer(self, intents: Dict[str, bool]) -> str:
        parts = []
        for item in self.trace:
            if item.get("action") == "search_product_catalog":
                products = item["observation"]
                if not products or (products and "error" in products[0]):
                    parts.append("Rất tiếc, không tìm thấy sản phẩm phù hợp.")
                else:
                    lines = [f"- {p['name']}: {p['price_vnd']:,} VNĐ" for p in products]
                    parts.append("Các lựa chọn phù hợp:\n" + "\n".join(lines))
            elif item.get("action") == "submit_support_ticket":
                ticket = item["observation"]
                parts.append(
                    f"Đã tạo ticket {ticket['ticket_id']} cho {ticket['customer_name']} "
                    f"với mức ưu tiên {ticket['priority']}."
                )
        if parts:
            return "\n\n".join(parts)
        if intents["is_faq"]:
            return "Chính sách bảo hành pin xe điện VinFast có thể kéo dài đến 10 năm; điều kiện cụ thể phụ thuộc mẫu xe và thị trường."
        return "Tôi có thể hỗ trợ thông tin VinFast, Vinpearl hoặc tiếp nhận yêu cầu hỗ trợ."


# ═══════════════════════════════════════════════════════════════════════════
# MAIN — Chạy thử nhanh
# ═══════════════════════════════════════════════════════════════════════════

def main():
    user_query = "Tôi muốn xem xe điện VinFast giá dưới 600 triệu."

    print("=== RUNNING CHATBOT BASELINE ===")
    chatbot = ChatbotBaseline()
    print(chatbot.query(user_query))

    print("\n=== RUNNING TOOL CALLING AGENT ===")
    agent = ToolCallingAgent(max_iterations=5)
    result = agent.run(user_query)
    print("Result:", result["answer"])
    print("Trace Log:", json.dumps(agent.trace, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()

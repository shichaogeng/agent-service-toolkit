import json
import requests
from typing import List, Dict, Any
from langchain.tools import BaseTool

class IndexMatchTool(BaseTool):
    name: str = "IndexMatch"
    description: str = "这是一个股票选指数函数，查询股票代码对应的相关度最高的指数。输入应为股票代码列表，例如：['600519.SH']"
    
    def _run(self, stock_codes: List[str]) -> Dict[str, Any]:
        url = "https://index.amcfortune.com/fundex-quote/smf/indexMatchResult"
        data = {"stockCodes": stock_codes}
        # 计算请求体的长度
        data_length = len(json.dumps(data))
        headers = {
            'Content-Type': 'application/json',
            'Host': 'index.amcfortune.com',
            'Content-Length': str(data_length),
            'Cookie': 'b63a17aefc9aa58a248373038abb902a=9b85c4d6b42df0e05f5939a58a7d186f'
        }
        
        # 打印HTTP请求信息
        print("Request URL:", url)
        print("Request Headers:", json.dumps(headers, ensure_ascii=False, indent=2))
        print("Request Data:", json.dumps(data, ensure_ascii=False, indent=2))
        
        try:
            response = requests.post(url, headers=headers, json=data)
            response.raise_for_status()
            result = response.json()
            
            print("API Response:", json.dumps(result, ensure_ascii=False, indent=2))  # 打印完整响应
            
            if result["code"] == "200" and "data" in result:
                # 提取items中的关键信息
                items = result["data"].get("items", [])[:3]  # 只取前5条数据
                print("Items:", json.dumps(items, ensure_ascii=False, indent=2))
                
                return {
                    "status": "success",
                    "message": "查询成功",
                    "data": items
                }
            else:
                return {
                    "status": "error",
                    "message": f"API返回错误: {result.get('msg', '未知错误')}",
                    "raw_response": result
                }
                
        except Exception as e:
            return {
                "status": "error",
                "message": f"请求失败: {str(e)}",
                "error": str(e)
            }

    async def _arun(self, stock_codes: List[str]) -> Dict[str, Any]:
        return self._run(stock_codes)
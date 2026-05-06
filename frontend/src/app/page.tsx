"use client";

import { useState, useEffect, useRef } from "react";
import { Send, Upload, FileText, Loader2, Search, CheckCircle2, AlertCircle } from "lucide-react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { motion, AnimatePresence } from "framer-motion";

interface Message {
  role: "user" | "assistant";
  content: string;
  sources?: any[];
  done?: boolean;
}

export default function ChatPage() {
  const [query, setQuery] = useState("");
  const [messages, setMessages] = useState<Message[]>([]);
  const [isStreaming, setIsStreaming] = useState(false);
  const [uploading, setUploading] = useState(false);
  const chatEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSearch = async () => {
    if (!query.trim() || isStreaming) return;

    const userMsg: Message = { role: "user", content: query };
    setMessages((prev) => [...prev, userMsg]);
    setQuery("");
    setIsStreaming(true);

    const assistantMsg: Message = { role: "assistant", content: "" };
    setMessages((prev) => [...prev, assistantMsg]);

    try {
      const response = await fetch("/api/search/stream", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: userMsg.content, history: [] }),
      });

      if (!response.body) throw new Error("No response body");

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let fullContent = "";

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value);
        const lines = chunk.split("\n");

        for (const line of lines) {
          if (line.startsWith("data: ")) {
            try {
              const data = JSON.parse(line.slice(6));
              if (data.type === "token") {
                fullContent += data.token;
                setMessages((prev) => {
                  const newMsgs = [...prev];
                  newMsgs[newMsgs.length - 1].content = fullContent;
                  return newMsgs;
                });
              } else if (data.type === "metadata") {
                setMessages((prev) => {
                  const newMsgs = [...prev];
                  newMsgs[newMsgs.length - 1].sources = data.sources;
                  return newMsgs;
                });
              } else if (data.type === "done") {
                setIsStreaming(false);
              }
            } catch (e) {
              console.error("Error parsing SSE chunk", e);
            }
          }
        }
      }
    } catch (error) {
      console.error("Search error", error);
      setIsStreaming(false);
    }
  };

  return (
    <div className="flex flex-col h-screen bg-slate-50 text-slate-900 font-sans">
      {/* Header */}
      <header className="border-b bg-white px-6 py-4 flex items-center justify-between shadow-sm">
        <div className="flex items-center gap-2">
          <div className="w-8 h-8 bg-brand-600 rounded-lg flex items-center justify-center text-white font-bold">D</div>
          <h1 className="text-xl font-bold tracking-tight">DocSearch <span className="text-brand-600">Pro</span></h1>
        </div>
        <div className="flex gap-4">
          <button className="flex items-center gap-2 px-3 py-1.5 rounded-md text-sm font-medium hover:bg-slate-100 transition-colors">
            <FileText size={16} />
            Documents
          </button>
        </div>
      </header>

      {/* Main Chat Area */}
      <main className="flex-1 overflow-y-auto p-4 md:p-8">
        <div className="max-w-4xl mx-auto space-y-6">
          {messages.length === 0 && (
            <div className="flex flex-col items-center justify-center py-20 text-center space-y-4">
              <div className="w-16 h-16 bg-brand-100 text-brand-600 rounded-full flex items-center justify-center mb-4">
                <Search size={32} />
              </div>
              <h2 className="text-3xl font-extrabold text-slate-800">Hỏi bất cứ điều gì về tài liệu của bạn</h2>
              <p className="text-slate-500 max-w-md">Tải lên tài liệu PDF, DOCX hoặc XLSX và nhận câu trả lời tức thì hoàn toàn ngoại tuyến.</p>
            </div>
          )}

          <AnimatePresence>
            {messages.map((msg, idx) => (
              <motion.div
                key={idx}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}
              >
                <div className={`max-w-[85%] rounded-2xl px-5 py-4 shadow-sm ${
                  msg.role === "user" 
                    ? "bg-brand-600 text-white" 
                    : "bg-white border border-slate-200"
                }`}>
                  <div className="prose prose-sm max-w-none">
                    <ReactMarkdown remarkPlugins={[remarkGfm]}>
                      {msg.content}
                    </ReactMarkdown>
                  </div>
                  
                  {msg.sources && msg.sources.length > 0 && (
                    <div className="mt-4 pt-4 border-t border-slate-100">
                      <p className="text-xs font-bold text-slate-400 uppercase mb-2">Nguồn trích dẫn</p>
                      <div className="flex flex-wrap gap-2">
                        {msg.sources.map((src, sIdx) => (
                          <div key={sIdx} className="px-2 py-1 bg-slate-100 rounded text-[10px] font-medium text-slate-600 border border-slate-200">
                            {src.file} {src.page ? `(Trang ${src.page})` : ""}
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              </motion.div>
            ))}
          </AnimatePresence>
          <div ref={chatEndRef} />
        </div>
      </main>

      {/* Input Area */}
      <footer className="border-t bg-white p-4">
        <div className="max-w-4xl mx-auto relative">
          <textarea
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && !e.shiftKey && (e.preventDefault(), handleSearch())}
            placeholder="Nhập câu hỏi của bạn..."
            className="w-full rounded-xl border border-slate-200 pr-24 pl-4 py-3 focus:outline-none focus:ring-2 focus:ring-brand-500 focus:border-transparent resize-none h-14"
          />
          <div className="absolute right-2 top-2 flex gap-2">
            <button 
              onClick={() => {}} 
              className="p-2 text-slate-400 hover:text-brand-600 hover:bg-slate-100 rounded-lg transition-colors"
              title="Upload files"
            >
              <Upload size={20} />
            </button>
            <button 
              onClick={handleSearch}
              disabled={!query.trim() || isStreaming}
              className={`p-2 rounded-lg transition-all ${
                query.trim() && !isStreaming 
                  ? "bg-brand-600 text-white shadow-md hover:bg-brand-700" 
                  : "bg-slate-100 text-slate-300 cursor-not-allowed"
              }`}
            >
              {isStreaming ? <Loader2 size={20} className="animate-spin" /> : <Send size={20} />}
            </button>
          </div>
        </div>
        <p className="text-[10px] text-center text-slate-400 mt-2">
          Hệ thống AI có thể nhầm lẫn. Hãy kiểm tra lại thông tin quan trọng.
        </p>
      </footer>
    </div>
  );
}

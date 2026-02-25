import React, { useState, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Send, Bot, User, Loader2, Sparkles, AlertCircle, Trash2 } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import { clsx } from 'clsx';
import { twMerge } from 'tailwind-merge';

function cn(...inputs) {
    return twMerge(clsx(inputs));
}

const App = () => {
    const [messages, setMessages] = useState(() => {
        const saved = localStorage.getItem('medichat_history');
        return saved ? JSON.parse(saved) : [
            { id: 1, role: 'bot', content: "Hello! I'm MediChat, your disease information assistant. How can I help you today?" }
        ];
    });
    const [input, setInput] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState(null);
    const messagesEndRef = useRef(null);

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    };

    useEffect(() => {
        localStorage.setItem('medichat_history', JSON.stringify(messages));
    }, [messages]);

    const clearHistory = () => {
        const initialMessage = [
            { id: 1, role: 'bot', content: "Hello! I'm MediChat, your disease information assistant. How can I help you today?" }
        ];
        setMessages(initialMessage);
        localStorage.setItem('medichat_history', JSON.stringify(initialMessage));
    };

    useEffect(() => {
        scrollToBottom();
    }, [messages]);

    const handleSend = async (e) => {
        e.preventDefault();
        if (!input.trim() || isLoading) return;

        const userMessage = { id: Date.now(), role: 'user', content: input };
        setMessages(prev => [...prev, userMessage]);
        setInput('');
        setIsLoading(true);
        setError(null);

        const history = messages.slice(1).map(msg => ({
            role: msg.role === 'bot' ? 'model' : 'user',
            content: msg.content
        }));

        try {
            const response = await fetch('http://localhost:8000/ask', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    question: input,
                    history: history
                })
            });

            if (!response.ok) {
                throw new Error('Failed to get response from the assistant.');
            }

            const data = await response.json();
            setMessages(prev => [...prev, { id: Date.now() + 1, role: 'bot', content: data.answer }]);
        } catch (err) {
            setError(err.message);
        } finally {
            setIsLoading(false);
        }
    };

    return (
        <div className="min-h-screen bg-[#f8fafc] text-[#0f172a] flex flex-col font-['Inter']">
            <header className="glass sticky top-0 z-10 px-6 py-4 flex items-center justify-between">
                <div className="flex items-center gap-2">
                    <div className="w-10 h-10 medical-gradient rounded-xl flex items-center justify-center text-white shadow-lg">
                        <Sparkles size={22} />
                    </div>
                    <div>
                        <h1 className="text-xl font-['Outfit'] font-bold text-medical-900 leading-tight">MediChat</h1>
                        <p className="text-xs text-medical-600 font-medium">Smart Health Assistant</p>
                    </div>
                </div>
                <div className="flex items-center gap-4">
                    <button
                        onClick={clearHistory}
                        className="p-2 text-slate-400 hover:text-red-500 transition-colors"
                        title="Clear conversation"
                    >
                        <Trash2 size={20} />
                    </button>
                    <div className="flex items-center gap-2 border-l pl-4 border-slate-200">
                        <div className="w-2 h-2 rounded-full bg-green-500 animate-pulse"></div>
                        <span className="text-sm font-medium text-slate-500">System Online</span>
                    </div>
                </div>
            </header>

            <main className="flex-1 max-w-4xl w-full mx-auto p-4 md:p-6 overflow-hidden flex flex-col">
                <div className="flex-1 overflow-y-auto space-y-6 pb-4 scrollbar-hide">
                    <AnimatePresence initial={false}>
                        {messages.map((message) => (
                            <motion.div
                                key={message.id}
                                initial={{ opacity: 0, y: 10, scale: 0.95 }}
                                animate={{ opacity: 1, y: 0, scale: 1 }}
                                className={cn(
                                    "flex items-start gap-3 max-w-[85%]",
                                    message.role === 'user' ? "ml-auto flex-row-reverse" : ""
                                )}
                            >
                                <div className={cn(
                                    "w-8 h-8 rounded-full flex items-center justify-center shrink-0",
                                    message.role === 'bot' ? "bg-medical-100 text-medical-700" : "bg-slate-200 text-slate-600"
                                )}>
                                    {message.role === 'bot' ? <Bot size={18} /> : <User size={18} />}
                                </div>
                                <div className={cn(
                                    "px-4 py-3 rounded-2xl text-sm leading-relaxed shadow-sm",
                                    message.role === 'bot'
                                        ? "glass text-slate-800 prose prose-slate prose-sm max-w-none"
                                        : "medical-gradient text-white"
                                )}>
                                    {message.role === 'bot' ? (
                                        <ReactMarkdown
                                            components={{
                                                ul: ({ node, ...props }) => <ul className="list-disc ml-4 space-y-1 my-2" {...props} />,
                                                ol: ({ node, ...props }) => <ol className="list-decimal ml-4 space-y-1 my-2" {...props} />,
                                                li: ({ node, ...props }) => <li className="marker:text-medical-500" {...props} />,
                                                p: ({ node, ...props }) => <p className="mb-2 last:mb-0" {...props} />,
                                                strong: ({ node, ...props }) => <strong className="font-bold text-medical-900" {...props} />
                                            }}
                                        >
                                            {message.content}
                                        </ReactMarkdown>
                                    ) : (
                                        message.content
                                    )}
                                </div>
                            </motion.div>
                        ))}
                    </AnimatePresence>

                    {isLoading && (
                        <motion.div
                            initial={{ opacity: 0 }}
                            animate={{ opacity: 1 }}
                            className="flex items-center gap-2 text-medical-600 ml-2"
                        >
                            <Loader2 size={16} className="animate-spin" />
                            <span className="text-xs font-medium">MediChat is thinking...</span>
                        </motion.div>
                    )}

                    {error && (
                        <div className="flex items-start gap-2 p-3 bg-red-50 text-red-600 rounded-xl text-sm border border-red-100">
                            <AlertCircle size={18} className="shrink-0 mt-0.5" />
                            <p>{error}</p>
                        </div>
                    )}
                    <div ref={messagesEndRef} />
                </div>
            </main>

            <div className="p-4 md:p-6 max-w-4xl w-full mx-auto">
                <form
                    onSubmit={handleSend}
                    className="relative group flex items-center bg-white rounded-2xl shadow-xl shadow-slate-200/50 border border-slate-100 p-2 transition-all duration-300 focus-within:ring-2 focus-within:ring-medical-500/20"
                >
                    <input
                        type="text"
                        value={input}
                        onChange={(e) => setInput(e.target.value)}
                        placeholder="Describe your symptoms or ask a health question..."
                        className="flex-1 bg-transparent py-3 px-4 outline-none text-slate-700 placeholder:text-slate-400"
                        disabled={isLoading}
                    />
                    <button
                        type="submit"
                        disabled={isLoading || !input.trim()}
                        className={cn(
                            "p-3 rounded-xl transition-all duration-200",
                            input.trim() && !isLoading
                                ? "medical-gradient text-white shadow-lg shadow-medical-500/30 hover:scale-105 active:scale-95"
                                : "bg-slate-100 text-slate-400"
                        )}
                    >
                        {isLoading ? <Loader2 size={20} className="animate-spin" /> : <Send size={20} />}
                    </button>
                </form>
                <p className="text-center text-[10px] text-slate-400 mt-3 uppercase tracking-widest font-bold">
                    Empowering Health with Knowledge • Be Informed
                </p>
            </div>
        </div>
    );
};

export default App;

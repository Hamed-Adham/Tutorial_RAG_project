"use client"

import React, { useState, useEffect, useRef } from "react"
import { Session, Message, SystemHealth, Citation, LatencyProfile, SelectedContextTurn } from "@/types"
import { Sidebar } from "@/components/Sidebar"
import { ChatHeader } from "@/components/ChatHeader"
import { ChatMessage } from "@/components/ChatMessage"
import { ChatInput } from "@/components/ChatInput"
import { ChapterModal } from "@/components/ChapterModal"
import { MetricsModal } from "@/components/MetricsModal"
import { EvaluationModal } from "@/components/EvaluationModal"
import { FAQCategoryModal } from "@/components/FAQCategoryModal"
import { Bot, Sparkles, BookOpen, HelpCircle } from "lucide-react"

export default function ChatPage() {
  // Session State
  const [sessions, setSessions] = useState<Session[]>([])
  const [activeSessionId, setActiveSessionId] = useState<string>("")
  const [messages, setMessages] = useState<Record<string, Message[]>>({})
  const [selectedChapters, setSelectedChapters] = useState<number[]>([])

  // Knowledge Source & Pipeline Mode State
  const [selectedSources, setSelectedSources] = useState<string[]>(["science_textbook"])
  const [pipelineMode, setPipelineMode] = useState<"advanced" | "naive">("advanced")
  const [selectedCategories, setSelectedCategories] = useState<string[]>([])

  const isAll = selectedSources.includes("science_textbook") && selectedSources.includes("administrative_faq")
  const isScienceActive = selectedSources.includes("science_textbook")
  const isFaqActive = selectedSources.includes("administrative_faq")
  const isFAQ = !isScienceActive && isFaqActive
  const source: "science_textbook" | "administrative_faq" | "all" = isAll
    ? "all"
    : isFaqActive
    ? "administrative_faq"
    : "science_textbook"

  // Conversational History & Context Selection State (User Story 6)
  const [historyEnabled, setHistoryEnabled] = useState<boolean>(true)
  const [selectedContextTurn, setSelectedContextTurn] = useState<SelectedContextTurn | null>(null)

  // Engine & UI State
  const [isStreaming, setIsStreaming] = useState<boolean>(true)
  const [topK, setTopK] = useState<number>(5)
  const [isLoading, setIsLoading] = useState<boolean>(false)
  const [isStreamingActive, setIsStreamingActive] = useState<boolean>(false)
  const [pipelineStepTitle, setPipelineStepTitle] = useState<string | null>(null)
  const [health, setHealth] = useState<SystemHealth | null>(null)

  // Modals & Panels
  const [sidebarOpen, setSidebarOpen] = useState<boolean>(true)
  const [chapterModalOpen, setChapterModalOpen] = useState<boolean>(false)
  const [faqModalOpen, setFaqModalOpen] = useState<boolean>(false)
  const [metricsModalOpen, setMetricsModalOpen] = useState<boolean>(false)
  const [evaluationModalOpen, setEvaluationModalOpen] = useState<boolean>(false)

  // Abort controller for streaming cancellation
  const abortControllerRef = useRef<AbortController | null>(null)
  const messagesEndRef = useRef<HTMLDivElement>(null)

  // Auto-scroll to bottom of messages
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages, activeSessionId, isStreamingActive])

  // Initialize or restore sessions from localStorage
  useEffect(() => {
    const saved = localStorage.getItem("unified_rag_sessions")
    if (saved) {
      try {
        const parsed = JSON.parse(saved)
        if (parsed.length > 0) {
          setSessions(parsed)
          setActiveSessionId(parsed[0].id)
          const savedMsgs = localStorage.getItem("unified_rag_messages")
          if (savedMsgs) {
            setMessages(JSON.parse(savedMsgs))
          }
        }
      } catch (e) {
        console.error("Failed to parse saved sessions", e)
      }
    } else {
      createNewSession()
    }

    // Restore pipeline mode and source preferences
    const savedMode = localStorage.getItem("unified_rag_pipeline_mode") as "advanced" | "naive" | null
    if (savedMode && (savedMode === "advanced" || savedMode === "naive")) {
      setPipelineMode(savedMode)
    }
    const savedSources = localStorage.getItem("unified_rag_sources")
    if (savedSources) {
      try {
        const parsed = JSON.parse(savedSources)
        if (Array.isArray(parsed) && parsed.length > 0) {
          setSelectedSources(parsed)
        }
      } catch (e) {
        // fallback
      }
    } else {
      const savedSource = localStorage.getItem("unified_rag_source") as "science_textbook" | "administrative_faq" | "all" | null
      if (savedSource === "administrative_faq") {
        setSelectedSources(["administrative_faq"])
      } else if (savedSource === "all") {
        setSelectedSources(["science_textbook", "administrative_faq"])
      }
    }
  }, [])

  const handlePipelineModeChange = (mode: "advanced" | "naive") => {
    setPipelineMode(mode)
    localStorage.setItem("unified_rag_pipeline_mode", mode)
    if (mode === "naive" && selectedSources.length > 1) {
      // In Naive mode, restrict to single source
      const single = [selectedSources[0]]
      setSelectedSources(single)
      localStorage.setItem("unified_rag_sources", JSON.stringify(single))
    }
  }

  const handleToggleSource = (src: "science_textbook" | "administrative_faq") => {
    if (pipelineMode === "advanced") {
      let nextSources: string[]
      if (selectedSources.includes(src)) {
        if (selectedSources.length > 1) {
          nextSources = selectedSources.filter((s) => s !== src)
        } else {
          // Cannot deselect the only active source
          return
        }
      } else {
        nextSources = [...selectedSources, src]
      }
      setSelectedSources(nextSources)
      localStorage.setItem("unified_rag_sources", JSON.stringify(nextSources))
    } else {
      const nextSources = [src]
      setSelectedSources(nextSources)
      localStorage.setItem("unified_rag_sources", JSON.stringify(nextSources))
    }
  }

  const handleSourceChange = (newSource: "science_textbook" | "administrative_faq" | "all") => {
    let nextSources: string[]
    if (newSource === "all") {
      nextSources = ["science_textbook", "administrative_faq"]
    } else {
      nextSources = [newSource]
    }
    setSelectedSources(nextSources)
    localStorage.setItem("unified_rag_sources", JSON.stringify(nextSources))
  }

  // Persist sessions and messages
  useEffect(() => {
    if (sessions.length > 0) {
      localStorage.setItem("unified_rag_sessions", JSON.stringify(sessions))
    }
    if (Object.keys(messages).length > 0) {
      localStorage.setItem("unified_rag_messages", JSON.stringify(messages))
    }
  }, [sessions, messages])

  // Fetch backend health periodically
  useEffect(() => {
    const checkHealth = async () => {
      try {
        const res = await fetch("/api/v1/health")
        if (res.ok) {
          const data = await res.json()
          setHealth(data)
        }
      } catch (e) {
        setHealth(null)
      }
    }
    checkHealth()
    const timer = setInterval(checkHealth, 10000)
    return () => clearInterval(timer)
  }, [])

  const createNewSession = () => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort()
      abortControllerRef.current = null
    }
    setIsLoading(false)
    setIsStreamingActive(false)
    setPipelineStepTitle(null)

    const newId = `sess_${Date.now().toString(36)}`
    const newSess: Session = {
      id: newId,
      title: "گفتگوی جدید",
      createdAt: new Date().toISOString(),
      updatedAt: new Date().toISOString(),
      selectedChapters: [],
    }
    setSessions((prev) => [newSess, ...prev])
    setActiveSessionId(newId)
    setSelectedChapters([])
    setSelectedContextTurn(null)
  }

  const handleSelectSession = (id: string) => {
    if (id !== activeSessionId) {
      if (abortControllerRef.current) {
        abortControllerRef.current.abort()
        abortControllerRef.current = null
      }
      setIsLoading(false)
      setIsStreamingActive(false)
      setPipelineStepTitle(null)
      setActiveSessionId(id)
      setSelectedContextTurn(null)
    }
  }

  const handleDeleteSession = async (id: string) => {
    try {
      await fetch(`/api/v1/sessions/${id}`, { method: "DELETE" })
    } catch (e) {
      // offline fallback
    }

    const remaining = sessions.filter((s) => s.id !== id)
    setSessions(remaining)

    const newMsgs = { ...messages }
    delete newMsgs[id]
    setMessages(newMsgs)

    if (activeSessionId === id) {
      if (remaining.length > 0) {
        setActiveSessionId(remaining[0].id)
      } else {
        createNewSession()
      }
    }
  }

  const handleClearCurrentSession = () => {
    if (!activeSessionId) return
    setMessages((prev) => ({
      ...prev,
      [activeSessionId]: [],
    }))
  }

  const handleSendMessage = async (queryText: string) => {
    if (!queryText.trim() || isLoading) return

    const userMsgId = `usr_${Date.now()}`
    const assistantMsgId = `ast_${Date.now()}`

    const userMsg: Message = {
      id: userMsgId,
      role: "user",
      content: queryText,
      timestamp: new Date().toISOString(),
    }

    const initialAssistantMsg: Message = {
      id: assistantMsgId,
      role: "assistant",
      content: "",
      timestamp: new Date().toISOString(),
    }

    // Append to messages
    setMessages((prev) => ({
      ...prev,
      [activeSessionId]: [...(prev[activeSessionId] || []), userMsg, initialAssistantMsg],
    }))

    // Update session title if first turn
    setSessions((prev) =>
      prev.map((s) => {
        if (s.id === activeSessionId && s.title === "گفتگوی جدید") {
          return { ...s, title: queryText.slice(0, 36) }
        }
        return s
      })
    )

    setIsLoading(true)
    const controller = new AbortController()
    abortControllerRef.current = controller

    // Prepare prior conversation turns (sliding window of last 4 turns) for memory continuity if history is enabled
    const priorTurns = historyEnabled
      ? (messages[activeSessionId] || [])
          .filter((m) => m.content && m.content.trim().length > 0)
          .slice(-4)
          .map((m) => ({
            role: m.role,
            content: m.content,
            turn_id: m.turn_id,
            q_and_a_summary: m.q_and_a_summary,
            summary: m.q_and_a_summary,
          }))
      : []

    const payload = {
      query: queryText,
      session_id: activeSessionId,
      pipeline_mode: pipelineMode,
      source: selectedSources[0],
      selected_sources: selectedSources,
      selected_chapters: selectedSources.includes("science_textbook") && selectedChapters.length > 0 ? selectedChapters : null,
      top_k: topK,
      stream: isStreaming,
      history: pipelineMode === "advanced" && historyEnabled ? priorTurns : undefined,
      history_enabled: pipelineMode === "advanced" ? historyEnabled : false,
      selected_context_turn_id: pipelineMode === "advanced" ? (selectedContextTurn?.id || undefined) : undefined,
    }

    // Reset selected context turn after dispatching request
    if (selectedContextTurn) {
      setSelectedContextTurn(null)
    }

    if (isStreaming) {
      // ── Server-Sent Events (SSE) Streaming Mode ──────────────────────────
      setIsStreamingActive(true)
      try {
        const resp = await fetch("/api/v1/chat/stream", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
          signal: controller.signal,
        })

        if (!resp.ok) {
          throw new Error(`HTTP ${resp.status}: ${resp.statusText}`)
        }

        const reader = resp.body?.getReader()
        if (!reader) throw new Error("No readable stream received")

        const decoder = new TextDecoder()
        let sseBuffer = ""
        let accumulatedContent = ""
        let finalCitations: Citation[] = []
        let finalProfile: LatencyProfile = {}
        let effectiveQuery = queryText
        let isRefusal = false
        let finalConversationalBadge: string | undefined = undefined

        let finalCost1k: any = undefined
        let finalTraceId: string | undefined = undefined
        let finalTurnId: string | undefined = undefined
        let finalQueryRewritten = false
        let finalSummary: string | undefined = undefined

        while (true) {
          const { done, value } = await reader.read()
          if (done) break

          sseBuffer += decoder.decode(value, { stream: true })
          const lines = sseBuffer.split("\n")
          sseBuffer = lines.pop() || ""

          for (const line of lines) {
            if (line.startsWith("data: ")) {
              const jsonStr = line.slice(6).trim()
              if (!jsonStr) continue
              try {
                const parsed = JSON.parse(jsonStr)

                // Pipeline Step Status event
                if (parsed.step && parsed.title) {
                  setPipelineStepTitle(parsed.title)
                }

                // Token event
                if (parsed.content) {
                  setPipelineStepTitle("نگارش پاسخ مستند...")
                  accumulatedContent += parsed.content
                  setMessages((prev) => {
                    const current = prev[activeSessionId] || []
                    return {
                      ...prev,
                      [activeSessionId]: current.map((m) =>
                        m.id === assistantMsgId ? { ...m, content: accumulatedContent } : m
                      ),
                    }
                  })
                }

                // Done event with metadata
                if (parsed.turn_id) {
                  finalTurnId = parsed.turn_id
                }
                if (parsed.query_rewritten !== undefined) {
                  finalQueryRewritten = parsed.query_rewritten
                }
                if (parsed.citations) {
                  finalCitations = parsed.citations
                }
                if (parsed.effective_query) {
                  effectiveQuery = parsed.effective_query
                }
                if (parsed.is_refusal !== undefined) {
                  isRefusal = parsed.is_refusal
                }
                if (parsed.cost_1k) {
                  finalCost1k = parsed.cost_1k
                }
                if (parsed.trace_id) {
                  finalTraceId = parsed.trace_id
                }
                if (parsed.conversational_context_badge) {
                  finalConversationalBadge = parsed.conversational_context_badge
                }
                if (parsed.q_and_a_summary || parsed.summary) {
                  finalSummary = parsed.q_and_a_summary || parsed.summary
                }
                if (parsed.latency_profile) {
                  finalProfile = {
                    ...parsed.latency_profile,
                    total_ms: parsed.total_ms || parsed.latency_profile.total_ms,
                  }
                } else if (parsed.ttft_ms || parsed.total_ms) {
                  finalProfile = {
                    total_ms: parsed.total_ms,
                    synthesis_ms: parsed.ttft_ms,
                  }
                }
              } catch (e) {
                // partial JSON frame
              }
            }
          }
        }

        // Finalize message object
        const resolvedContent = accumulatedContent.trim()
          ? accumulatedContent
          : (isRefusal
              ? "این پرسش خارج از چارچوب کتاب علوم تجربی هفتم است."
              : "پاسخی از سرور دریافت نشد یا ارتباط با مدل محلی قطع شد.")

        setMessages((prev) => {
          const current = prev[activeSessionId] || []
          return {
            ...prev,
            [activeSessionId]: current.map((m) => {
              if (m.id === assistantMsgId) {
                return {
                  ...m,
                  content: resolvedContent,
                  citations: finalCitations,
                  latency_profile: finalProfile,
                  effective_query: effectiveQuery,
                  query_rewritten: finalQueryRewritten,
                  is_refusal: isRefusal,
                  conversational_context_badge: finalConversationalBadge,
                  cost_1k: finalCost1k,
                  trace_id: finalTraceId,
                  turn_id: finalTurnId,
                  q_and_a_summary: finalSummary,
                }
              }
              if (m.id === userMsgId && finalTurnId) {
                return {
                  ...m,
                  turn_id: finalTurnId,
                }
              }
              return m
            }),
          }
        })
      } catch (err: any) {
        if (err.name !== "AbortError") {
          setMessages((prev) => {
            const current = prev[activeSessionId] || []
            return {
              ...prev,
              [activeSessionId]: current.map((m) =>
                m.id === assistantMsgId
                  ? {
                      ...m,
                      content: `خطا در دریافت پاسخ: ${err.message || "پاسخ از سرور دریافت نشد."}`,
                    }
                  : m
              ),
            }
          })
        }
      } finally {
        setIsLoading(false)
        setIsStreamingActive(false)
        abortControllerRef.current = null
      }
    } else {
      // ── Synchronous Batch REST Mode ───────────────────────────────────────
      try {
        const resp = await fetch("/api/v1/chat", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
          signal: controller.signal,
        })

        if (!resp.ok) {
          throw new Error(`HTTP ${resp.status}: ${resp.statusText}`)
        }

        const data = await resp.json()

        setMessages((prev) => {
          const current = prev[activeSessionId] || []
          return {
            ...prev,
            [activeSessionId]: current.map((m) => {
              if (m.id === assistantMsgId) {
                return {
                  ...m,
                  content: data.answer,
                  citations: data.citations,
                  latency_profile: data.latency_profile,
                  effective_query: data.effective_query,
                  query_rewritten: data.query_rewritten,
                  is_refusal: data.is_refusal,
                  conversational_context_badge: data.conversational_context_badge,
                  trace_id: data.trace_id,
                  cost_1k: data.cost_1k,
                  turn_id: data.turn_id,
                  q_and_a_summary: data.q_and_a_summary,
                }
              }
              if (m.id === userMsgId && data.turn_id) {
                return {
                  ...m,
                  turn_id: data.turn_id,
                }
              }
              return m
            }),
          }
        })
      } catch (err: any) {
        if (err.name !== "AbortError") {
          setMessages((prev) => {
            const current = prev[activeSessionId] || []
            return {
              ...prev,
              [activeSessionId]: current.map((m) =>
                m.id === assistantMsgId
                  ? {
                      ...m,
                      content: `خطا در ارتباط با سامانه: ${err.message}`,
                    }
                  : m
              ),
            }
          })
        }
      } finally {
        setIsLoading(false)
        setIsStreamingActive(false)
        setPipelineStepTitle(null)
        abortControllerRef.current = null
      }
    }
  }

  const handleStopStreaming = () => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort()
      setIsLoading(false)
      setIsStreamingActive(false)
      setPipelineStepTitle(null)
    }
  }

  const currentMessages = messages[activeSessionId] || []
  const activeSession = sessions.find((s) => s.id === activeSessionId) || null

  return (
    <div className="flex h-screen w-screen overflow-hidden bg-background">
      {/* Collapsible Left Sidebar */}
      <Sidebar
        sessions={sessions}
        activeSessionId={activeSessionId}
        onSelectSession={handleSelectSession}
        onNewSession={createNewSession}
        onDeleteSession={handleDeleteSession}
        onOpenEvaluationModal={() => setEvaluationModalOpen(true)}
        health={health}
        isOpen={sidebarOpen}
        onToggleOpen={() => setSidebarOpen(!sidebarOpen)}
      />

      {/* Main Chat Area */}
      <main className="flex-1 flex flex-col h-screen overflow-hidden relative">
        {/* Top Header */}
        <ChatHeader
          session={activeSession}
          selectedChapters={selectedChapters}
          selectedCategories={selectedCategories}
          onOpenChapterModal={() => setChapterModalOpen(true)}
          onOpenScopeModal={() => setChapterModalOpen(true)}
          onOpenMetricsModal={() => setMetricsModalOpen(true)}
          onClearSession={handleClearCurrentSession}
          isStreaming={isStreaming}
          onToggleStreaming={() => setIsStreaming(!isStreaming)}
          topK={topK}
          onChangeTopK={setTopK}
          health={health}
          historyEnabled={historyEnabled}
          onToggleHistory={() => setHistoryEnabled(!historyEnabled)}
          pipelineMode={pipelineMode}
          onChangePipelineMode={handlePipelineModeChange}
          source={source}
          selectedSources={selectedSources}
          onToggleSource={handleToggleSource}
          onChangeSource={handleSourceChange}
        />

        {/* Message Scroll Container */}
        <div className="flex-1 overflow-y-auto">
          {currentMessages.length === 0 ? (
            /* Empty State Hero */
            <div className="h-full flex flex-col items-center justify-center p-6 text-center max-w-xl mx-auto space-y-4">
              <div className={`w-14 h-14 rounded-2xl flex items-center justify-center shadow-lg border ${
                isAll
                  ? "bg-indigo-500/10 text-indigo-600 dark:text-indigo-400 border-indigo-500/20"
                  : isFAQ
                  ? "bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 border-emerald-500/20"
                  : "bg-primary/10 text-primary border-primary/20"
              }`}>
                {isAll ? <Sparkles className="w-7 h-7" /> : isFAQ ? <HelpCircle className="w-7 h-7" /> : <Bot className="w-7 h-7" />}
              </div>
              <div className="space-y-1.5">
                <h1 className="text-lg md:text-xl font-bold tracking-tight">
                  {isAll
                    ? "سامانه جامع چندمنبعی (کتاب علوم + پرسش‌های متداول)"
                    : isFAQ
                    ? "سامانه پاسخگویی به پرسش‌های متداول (صندوق نوآوری و شکوفایی)"
                    : "دستیار هوشمند علوم تجربی پایه هفتم"}
                </h1>
                <p className="text-xs md:text-sm text-muted-foreground leading-relaxed">
                  {isAll
                    ? "مسیریابی هوشمند پرسش و استخراج متنی و معنایی همزمان از سرفصل‌های علوم تجربی هفتم و ضوابط صندوق نوآوری"
                    : isFAQ
                    ? "پاسخ به سوالات ضوابط، تسهیلات، ضمانت‌نامه‌ها، لیزینگ، صکوک و فرآیندهای پذیرش با دسترسی به ۱۳۴ پرسش متداول رسمی"
                    : "پاسخگویی به سوالات، تحلیل آزمایش‌ها، و بررسی مفاهیم علمی ۱۵ فصل کتاب درسی با استناد دقیق به شماره صفحات"}
                </p>
              </div>

              {/* Mode Badge Indicator */}
              <button
                onClick={() => setChapterModalOpen(true)}
                className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full border border-border bg-muted/40 hover:bg-muted text-xs text-muted-foreground hover:text-foreground transition-all shadow-xs"
              >
                {isAll ? (
                  <>
                    <Sparkles className="w-3.5 h-3.5 text-primary" />
                    <span>
                      دامنه: ۲ منبع فعال ({selectedChapters.length === 0 ? "کل ۱۵ فصل کتاب" : `${selectedChapters.length} فصل`} + ۱۳۴ پرسش متداول)
                    </span>
                  </>
                ) : isFAQ ? (
                  <>
                    <HelpCircle className="w-3.5 h-3.5 text-emerald-600 dark:text-emerald-400" />
                    <span>دامنه جستجو: کل بانک پرسش‌های متداول (۱۳۴ پرسش)</span>
                  </>
                ) : (
                  <>
                    <BookOpen className="w-3.5 h-3.5 text-primary" />
                    <span>
                      {selectedChapters.length === 0
                        ? "دامنه جستجو: تمام ۱۵ فصل کتاب"
                        : `دامنه تمرین: فصل‌های ${selectedChapters.join("، ")}`}
                    </span>
                  </>
                )}
              </button>
            </div>
          ) : (
            /* Render Messages List */
            currentMessages.map((msg, idx) => (
              <ChatMessage
                key={msg.id}
                message={msg}
                isStreaming={isStreamingActive && idx === currentMessages.length - 1}
                streamingStatus={idx === currentMessages.length - 1 ? pipelineStepTitle : null}
                onSelectAsContext={(selectedMsg) =>
                  setSelectedContextTurn({
                    id: selectedMsg.id,
                    content: selectedMsg.content,
                    role: selectedMsg.role,
                  })
                }
              />
            ))
          )}
          <div ref={messagesEndRef} />
        </div>

        {/* Floating Chat Input Box */}
        <ChatInput
          onSendMessage={handleSendMessage}
          onStopStreaming={handleStopStreaming}
          isLoading={isLoading}
          isStreaming={isStreamingActive}
          source={source}
          selectedSources={selectedSources}
          selectedChapters={selectedChapters}
          onOpenChapterModal={() => setChapterModalOpen(true)}
          onOpenScopeModal={() => setChapterModalOpen(true)}
          selectedContextTurn={selectedContextTurn}
          onClearSelectedContext={() => setSelectedContextTurn(null)}
        />
      </main>

      {/* Scope & Chapter Selection Modal */}
      <ChapterModal
        isOpen={chapterModalOpen}
        onClose={() => setChapterModalOpen(false)}
        selectedChapters={selectedChapters}
        onSaveChapters={setSelectedChapters}
        selectedSources={selectedSources}
        onToggleSource={handleToggleSource}
        pipelineMode={pipelineMode}
      />

      {/* FAQ Category Selection Modal */}
      <FAQCategoryModal
        isOpen={faqModalOpen}
        onClose={() => setFaqModalOpen(false)}
        selectedCategories={selectedCategories}
        onSaveCategories={setSelectedCategories}
      />

      {/* Production Telemetry & Metrics Modal */}
      <MetricsModal isOpen={metricsModalOpen} onClose={() => setMetricsModalOpen(false)} />

      {/* Multi-Suite Benchmark & Evaluation Dashboard Modal (Feature 007) */}
      <EvaluationModal
        isOpen={evaluationModalOpen}
        onClose={() => setEvaluationModalOpen(false)}
      />
    </div>
  )
}

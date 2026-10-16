"use client"

import React, { useState } from "react"
import { Session, SystemHealth } from "@/types"
import {
  Plus,
  MessageSquare,
  Trash2,
  BookOpen,
  Activity,
  Moon,
  Sun,
  PanelLeftClose,
  PanelLeft,
  ChevronLeft,
  Scale
} from "lucide-react"

interface SidebarProps {
  sessions: Session[]
  activeSessionId: string
  onSelectSession: (id: string) => void
  onNewSession: () => void
  onDeleteSession: (id: string) => void
  onOpenChapterModal?: () => void
  onOpenEvaluationModal?: () => void
  selectedChapters?: number[]
  health: SystemHealth | null
  isOpen: boolean
  onToggleOpen: () => void
}

export function Sidebar({
  sessions,
  activeSessionId,
  onSelectSession,
  onNewSession,
  onDeleteSession,
  onOpenChapterModal,
  onOpenEvaluationModal,
  selectedChapters = [],
  health,
  isOpen,
  onToggleOpen,
}: SidebarProps) {
  const [isDark, setIsDark] = useState(true)

  const toggleTheme = () => {
    setIsDark(!isDark)
    if (document.documentElement.classList.contains("dark")) {
      document.documentElement.classList.remove("dark")
    } else {
      document.documentElement.classList.add("dark")
    }
  }

  if (!isOpen) {
    return (
      <div className="p-3 border-l border-border bg-card/60 backdrop-blur-md flex flex-col justify-between items-center z-20">
        <button
          onClick={onToggleOpen}
          className="p-2 rounded-lg hover:bg-muted text-muted-foreground hover:text-foreground transition-colors"
          title="باز کردن منو"
        >
          <PanelLeft className="w-5 h-5" />
        </button>
        <button
          onClick={onNewSession}
          className="p-2 rounded-lg bg-primary/10 text-primary hover:bg-primary/20 transition-colors mt-4"
          title="گفتگوی جدید"
        >
          <Plus className="w-5 h-5" />
        </button>
        {onOpenEvaluationModal && (
          <button
            onClick={onOpenEvaluationModal}
            className="p-2 rounded-lg bg-indigo-500/10 text-indigo-400 hover:bg-indigo-500/20 transition-colors mt-2"
            title="ارزیابی پایپ‌لاین (Benchmarks)"
          >
            <Scale className="w-5 h-5" />
          </button>
        )}
      </div>
    )
  }

  return (
    <aside className="w-72 md:w-80 border-l border-border bg-card/80 backdrop-blur-lg flex flex-col h-screen select-none transition-all duration-300 z-20">
      {/* Header & New Chat */}
      <div className="p-3.5 border-b border-border/60 flex items-center justify-between gap-2">
        <button
          onClick={onNewSession}
          className="flex-1 flex items-center justify-center gap-2 bg-primary text-primary-foreground font-medium py-2 px-3 rounded-xl hover:opacity-95 active:scale-[0.98] transition-all text-sm shadow-sm"
        >
          <Plus className="w-4 h-4" />
          <span>گفتگوی جدید</span>
        </button>
        <button
          onClick={onToggleOpen}
          className="p-2 rounded-xl hover:bg-muted text-muted-foreground hover:text-foreground transition-colors"
          title="بستن منو"
        >
          <PanelLeftClose className="w-4 h-4" />
        </button>
      </div>

      {/* Evaluation Dashboard Quick Launcher (Mentor Feature) */}
      {onOpenEvaluationModal && (
        <div className="px-3 pt-3">
          <button
            onClick={onOpenEvaluationModal}
            className="w-full flex items-center justify-between p-2.5 rounded-xl border border-primary/30 bg-primary/10 hover:bg-primary/20 active:scale-[0.98] transition-all text-right shadow-sm group"
          >
            <div className="flex items-center gap-2.5">
              <div className="w-7 h-7 rounded-lg bg-primary/20 text-primary flex items-center justify-center shrink-0 group-hover:scale-105 transition-transform">
                <Scale className="w-4 h-4" />
              </div>
              <div>
                <div className="text-xs font-bold text-foreground">ارزیابی پایپ‌لاین</div>
                <div className="text-[10px] text-muted-foreground">پنل بنچ‌مارک و سنجه‌های منتور</div>
              </div>
            </div>
            <span className="text-[10px] font-mono font-bold bg-primary text-primary-foreground px-1.5 py-0.5 rounded-md">
              Evals
            </span>
          </button>
        </div>
      )}

      {/* Sessions List */}
      <div className="flex-1 overflow-y-auto p-2 space-y-1">
        <div className="text-[11px] font-semibold text-muted-foreground px-2 py-1 uppercase tracking-wider">
          تاریخچه گفتگوها
        </div>
        {sessions.length === 0 ? (
          <div className="text-center py-8 text-xs text-muted-foreground">
            هنوز گفتگویی ثبت نشده است.
          </div>
        ) : (
          sessions.map((sess) => {
            const isActive = sess.id === activeSessionId
            return (
              <div
                key={sess.id}
                onClick={() => onSelectSession(sess.id)}
                className={`group flex items-center justify-between p-2.5 rounded-xl text-xs cursor-pointer transition-all ${
                  isActive
                    ? "bg-primary/15 text-primary font-medium"
                    : "hover:bg-muted/60 text-muted-foreground hover:text-foreground"
                }`}
              >
                <div className="flex items-center gap-2.5 truncate">
                  <MessageSquare className={`w-3.5 h-3.5 shrink-0 ${isActive ? "text-primary" : "text-muted-foreground"}`} />
                  <span className="truncate">{sess.title || "گفتگوی بدون عنوان"}</span>
                </div>
                <button
                  onClick={(e) => {
                    e.stopPropagation()
                    onDeleteSession(sess.id)
                  }}
                  className="opacity-0 group-hover:opacity-100 p-1 hover:text-destructive hover:bg-destructive/10 rounded-lg transition-all"
                  title="حذف گفتگو"
                >
                  <Trash2 className="w-3.5 h-3.5" />
                </button>
              </div>
            )
          })
        )}
      </div>

      {/* Footer: Health & Theme */}
      <div className="p-3 border-t border-border/60 bg-muted/20 space-y-2">
        {/* Backend Health Badge */}
        <div className="flex items-center justify-between text-[11px] px-1 text-muted-foreground">
          <div className="flex items-center gap-1.5">
            <span
              className={`w-2 h-2 rounded-full ${
                health?.status === "healthy"
                  ? "bg-emerald-500 shadow-[0_0_8px_rgba(16,185,129,0.5)]"
                  : health?.status === "degraded"
                  ? "bg-emerald-400 shadow-[0_0_6px_rgba(52,211,153,0.4)]"
                  : health?.status === "unhealthy"
                  ? "bg-rose-500 shadow-[0_0_8px_rgba(244,63,94,0.5)]"
                  : "bg-amber-500 animate-pulse"
              }`}
            />
            <span>وضعیت هسته:</span>
          </div>
          <span className="font-mono text-[10px]">
            {health?.status === "healthy"
              ? "متصل (کامل)"
              : health?.status === "degraded"
              ? "متصل (مستقل / BM25)"
              : health?.status === "unhealthy"
              ? "قطع ارتباط"
              : "درحال بررسی..."}
          </span>
        </div>

        {/* Theme toggle & repo link */}
        <div className="flex items-center justify-between pt-1">
          <button
            onClick={toggleTheme}
            className="flex items-center gap-2 p-1.5 rounded-lg hover:bg-muted text-xs text-muted-foreground hover:text-foreground transition-colors"
          >
            {isDark ? <Sun className="w-4 h-4" /> : <Moon className="w-4 h-4" />}
            <span>{isDark ? "حالت روشن" : "حالت تاریک"}</span>
          </button>
          <div className="text-[10px] text-muted-foreground font-mono">Unified RAG v1.0</div>
        </div>
      </div>
    </aside>
  )
}

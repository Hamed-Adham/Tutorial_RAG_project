"use client"

import React, { useState, useEffect } from "react"
import {
  X,
  Play,
  RotateCcw,
  CheckCircle2,
  AlertTriangle,
  Download,
  ExternalLink,
  Sliders,
  Target,
  Sparkles,
  Layers,
  Search,
  Scale,
  Brain,
  Timer,
  Coins,
  ShieldCheck,
  ChevronDown,
} from "lucide-react"
import {
  EvaluationRunRequest,
  EvaluationRunStatus,
  EvaluationScope,
  SplitType,
  SuiteInfo,
  EvaluationReport,
} from "@/types"

interface EvaluationModalProps {
  isOpen: boolean
  onClose: () => void
}

const DEFAULT_SUITES: SuiteInfo[] = [
  {
    id: "all",
    name: "همه سوئیت‌ها (جامع)",
    description: "ارزیابی کامل ۳۶۰ مورد در تمام ۵ دامنه کالیبره شده",
    total_items: 360,
    dev_items: 90,
    test_items: 270,
    difficulty_profile: "۲۰٪ آسان، ۵۰٪ متوسط، ۳۰٪ دشوار",
  },
  {
    id: "science_curriculum",
    name: "علوم تجربی هفتم",
    description: "۱۵ فصل کتاب درسی با استدلال چندمرحله‌ای، نمودار و آزمایش",
    total_items: 120,
    dev_items: 30,
    test_items: 90,
    difficulty_profile: "زیست، شیمی، فیزیک، زمین (20% E, 50% M, 30% H)",
  },
  {
    id: "administrative_faq",
    name: "قوانین و آیین‌نامه‌ها",
    description: "پرسمان‌های اداری، تلاقی اصطلاحات و ابلاغیه‌های چندشرطی",
    total_items: 80,
    dev_items: 20,
    test_items: 60,
    difficulty_profile: "ادبیات اداری و بانکی (20% E, 50% M, 30% H)",
  },
  {
    id: "conversational_memory",
    name: "حافظه مکالمه",
    description: "گفت‌وگوهای ۲ تا ۱۰ نوبته، ارجاع ضمایر و تغییر مبحث",
    total_items: 60,
    dev_items: 15,
    test_items: 45,
    difficulty_profile: "پیگیری مرجع ضمیر (20% E, 50% M, 30% H)",
  },
  {
    id: "subquery_decomposition",
    name: "شکست پرسمان",
    description: "پرسش‌های مرکب نیازمند تفکیک به ۲ تا ۳ زیرپرسش",
    total_items: 50,
    dev_items: 12,
    test_items: 38,
    difficulty_profile: "پرسش‌های چندبخشی (20% E, 50% M, 30% H)",
  },
  {
    id: "adversarial_refusal",
    name: "عدم‌پاسخ و امنیت",
    description: "پرسمان‌های نامرتبط، فرضیات غلط، تزریق پرامپت و خدمات تجاری",
    total_items: 50,
    dev_items: 13,
    test_items: 37,
    difficulty_profile: "بررسی فرمول قطعی عدم پاسخ (100% Adversarial)",
  },
]

export function EvaluationModal({ isOpen, onClose }: EvaluationModalProps) {
  // Config state
  const [selectedSuite, setSelectedSuite] = useState<string>("science_curriculum")
  const [split, setSplit] = useState<SplitType>("dev")
  const [scope, setScope] = useState<EvaluationScope>("end_to_end")
  const [judgeModel, setJudgeModel] = useState<string>("subq")
  const [limit, setLimit] = useState<number>(10)

  // Suites from API or fallback
  const [suites, setSuites] = useState<SuiteInfo[]>(DEFAULT_SUITES)

  // Execution state
  const [currentRunId, setCurrentRunId] = useState<string | null>(null)
  const [runStatus, setRunStatus] = useState<EvaluationRunStatus | null>(null)
  const [isStarting, setIsStarting] = useState(false)
  const [error, setError] = useState<string | null>(null)

  // Active tab in results: "scorecard" | "matrix" | "failures"
  const [activeTab, setActiveTab] = useState<"scorecard" | "matrix" | "failures">("scorecard")

  // Load suites on mount
  useEffect(() => {
    if (isOpen) {
      fetch("/api/v1/evaluation/suites")
        .then((res) => res.json())
        .then((data) => {
          if (data?.suites && Array.isArray(data.suites)) {
            setSuites(data.suites)
          }
        })
        .catch(() => {
          // Use default fallback
        })
    }
  }, [isOpen])

  // Poll status when a run is active
  useEffect(() => {
    let timer: NodeJS.Timeout | null = null

    if (currentRunId && (runStatus?.status === "queued" || runStatus?.status === "running")) {
      timer = setInterval(async () => {
        try {
          const res = await fetch(`/api/v1/evaluation/status/${currentRunId}`)
          if (!res.ok) return
          const data: EvaluationRunStatus = await res.json()
          setRunStatus(data)
          if (data.status === "completed" || data.status === "failed") {
            if (timer) clearInterval(timer)
          }
        } catch (e) {
          console.error("Polling error:", e)
        }
      }, 1500)
    }

    return () => {
      if (timer) clearInterval(timer)
    }
  }, [currentRunId, runStatus?.status])

  // Start Benchmark Run
  const handleStartRun = async () => {
    setIsStarting(true)
    setError(null)

    const payload: EvaluationRunRequest = {
      suite: selectedSuite === "all" ? null : (selectedSuite as any),
      split,
      scope,
      judge_model: judgeModel,
      limit,
    }

    try {
      const res = await fetch("/api/v1/evaluation/run", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      })

      if (!res.ok) {
        const errJson = await res.json().catch(() => ({}))
        throw new Error(errJson.detail || `HTTP ${res.status}`)
      }

      const data = await res.json()
      setCurrentRunId(data.run_id)
      setRunStatus({
        run_id: data.run_id,
        status: "running",
        progress_percentage: 0,
        total_items: limit,
        processed_items: 0,
        elapsed_seconds: 0,
      })
    } catch (e: any) {
      setError(e.message || "Failed to start evaluation run")
    } finally {
      setIsStarting(false)
    }
  }

  // Download Markdown Report
  const handleDownloadMarkdown = async () => {
    if (!currentRunId) return
    try {
      const res = await fetch(`/api/v1/evaluation/report/${currentRunId}/markdown`)
      if (!res.ok) throw new Error("Report not ready")
      const text = await res.text()

      const blob = new Blob([text], { type: "text/markdown;charset=utf-8" })
      const url = URL.createObjectURL(blob)
      const a = document.createElement("a")
      a.href = url
      a.download = `evaluation_report_${currentRunId}.md`
      document.body.appendChild(a)
      a.click()
      document.body.removeChild(a)
      URL.revokeObjectURL(url)
    } catch (e: any) {
      alert("خطا در دریافت فایل گزارش: " + e.message)
    }
  }

  if (!isOpen) return null

  const selectedSuiteInfo = suites.find((s) => s.id === selectedSuite) || suites[0]
  const report: EvaluationReport | undefined = runStatus?.report
  const isRunning = runStatus?.status === "running" || runStatus?.status === "queued"
  const isDone = runStatus?.status === "completed"

  return (
    <div className="fixed inset-0 bg-background/80 backdrop-blur-md z-50 flex items-center justify-center p-3 sm:p-5">
      <div className="bg-card border border-border rounded-2xl max-w-4xl w-full max-h-[92vh] flex flex-col shadow-2xl overflow-hidden animate-in zoom-in-95 duration-150">
        {/* Header */}
        <div className="p-4 sm:px-6 border-b border-border/80 flex items-center justify-between bg-card/90">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-xl bg-primary/10 text-primary flex items-center justify-center shadow-inner">
              <Scale className="w-5 h-5" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h2 className="text-base font-bold text-foreground">
                  پنل ارزیابی زنده پایپ‌لاین و بنچ‌مارک‌ها
                </h2>
                <span className="text-[10px] font-mono bg-primary/15 text-primary px-2 py-0.5 rounded-full font-semibold border border-primary/20">
                  Feature 007
                </span>
              </div>
              <p className="text-xs text-muted-foreground font-sans mt-0.5">
                داور تکمیلی SubQ Decisions • ماتریس تشخیصی ۲×۲ • گیت‌های قطعی Persian
              </p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-2 hover:bg-muted text-muted-foreground hover:text-foreground rounded-xl transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Scrollable Container */}
        <div className="flex-1 overflow-y-auto p-4 sm:p-6 space-y-6">
          {/* Controls Bar */}
          <div className="bg-muted/30 border border-border/70 rounded-2xl p-4 space-y-4">
            {/* Suite Tabs */}
            <div>
              <div className="text-xs font-semibold text-muted-foreground mb-2 flex items-center gap-1.5">
                <Layers className="w-3.5 h-3.5" />
                <span>انتخاب سوئیت بنچ‌مارک (Suite Selector):</span>
              </div>
              <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-6 gap-2">
                {suites.map((s) => (
                  <button
                    key={s.id}
                    onClick={() => setSelectedSuite(s.id)}
                    className={`px-2.5 py-2 rounded-xl text-xs font-medium text-center transition-all border ${
                      selectedSuite === s.id
                        ? "bg-primary text-primary-foreground border-primary shadow-sm scale-[1.02]"
                        : "bg-card/70 hover:bg-card text-muted-foreground hover:text-foreground border-border/60"
                    }`}
                  >
                    <div className="truncate font-semibold">{s.name}</div>
                    <div className="text-[10px] opacity-75 mt-0.5 font-mono">
                      {s.total_items} پرسش
                    </div>
                  </button>
                ))}
              </div>
              {selectedSuiteInfo && (
                <div className="mt-2.5 px-3 py-1.5 rounded-lg bg-card/60 border border-border/40 text-[11px] text-muted-foreground flex flex-wrap items-center justify-between gap-2">
                  <span>{selectedSuiteInfo.description}</span>
                  <span className="font-mono text-primary font-medium">
                    {selectedSuiteInfo.difficulty_profile}
                  </span>
                </div>
              )}
            </div>

            {/* Execution Options Grid */}
            <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-3 pt-1 border-t border-border/50">
              {/* Scope */}
              <div>
                <label className="text-[11px] font-semibold text-muted-foreground block mb-1">
                  حالت اجرا (Scope):
                </label>
                <select
                  value={scope}
                  onChange={(e) => setScope(e.target.value as EvaluationScope)}
                  disabled={isRunning}
                  className="w-full bg-card border border-border rounded-xl px-3 py-1.5 text-xs focus:ring-1 focus:ring-primary outline-none"
                >
                  <option value="end_to_end">پایپ‌لاین کامل (End-to-End)</option>
                  <option value="retrieve_only">فقط بازیابی (Retrieve Only)</option>
                  <option value="gates_only">فقط گیت‌های قطعی (Gates Only)</option>
                  <option value="synthesis_only">فقط سنتز (Synthesis Only)</option>
                </select>
              </div>

              {/* Split */}
              <div>
                <label className="text-[11px] font-semibold text-muted-foreground block mb-1">
                  مجموعه داده (Split):
                </label>
                <select
                  value={split}
                  onChange={(e) => setSplit(e.target.value as SplitType)}
                  disabled={isRunning}
                  className="w-full bg-card border border-border rounded-xl px-3 py-1.5 text-xs focus:ring-1 focus:ring-primary outline-none"
                >
                  <option value="dev">توسعه Dev (۲۵٪ کالیبره)</option>
                  <option value="test">آزمون Test (۷۵٪ قفل‌شده)</option>
                </select>
              </div>

              {/* Judge Model */}
              <div>
                <label className="text-[11px] font-semibold text-muted-foreground block mb-1">
                  مدل داور (LLM Judge):
                </label>
                <select
                  value={judgeModel}
                  onChange={(e) => setJudgeModel(e.target.value)}
                  disabled={isRunning}
                  className="w-full bg-card border border-border rounded-xl px-3 py-1.5 text-xs focus:ring-1 focus:ring-primary outline-none"
                >
                  <option value="subq">SubQ Decisions (subq-snap)</option>
                  <option value="jev">TypeSafe Jev (System One)</option>
                  <option value="gemini_llm">Gemini 2.5 (CoT Local)</option>
                </select>
              </div>

              {/* Limit */}
              <div>
                <label className="text-[11px] font-semibold text-muted-foreground block mb-1">
                  تعداد موارد (Limit):
                </label>
                <div className="flex items-center gap-2">
                  <input
                    type="number"
                    min={1}
                    max={100}
                    value={limit}
                    onChange={(e) => setLimit(Math.max(1, parseInt(e.target.value) || 1))}
                    disabled={isRunning}
                    className="w-full bg-card border border-border rounded-xl px-3 py-1.5 text-xs font-mono focus:ring-1 focus:ring-primary outline-none"
                  />
                  <div className="flex gap-1">
                    {[5, 10, 25].map((v) => (
                      <button
                        key={v}
                        onClick={() => setLimit(v)}
                        disabled={isRunning}
                        className={`px-2 py-1 text-[10px] rounded-lg border font-mono ${
                          limit === v
                            ? "bg-primary/20 text-primary border-primary/40 font-bold"
                            : "hover:bg-muted border-border/60 text-muted-foreground"
                        }`}
                      >
                        {v}
                      </button>
                    ))}
                  </div>
                </div>
              </div>
            </div>

            {/* Launch Action */}
            <div className="flex items-center justify-between pt-2">
              <div className="text-xs text-muted-foreground flex items-center gap-2">
                <span className="w-2 h-2 rounded-full bg-emerald-500" />
                <span>
                  آماده اجرا روی{" "}
                  <strong className="text-foreground font-mono">{limit}</strong> پرسش از سوئیت{" "}
                  <strong className="text-foreground">{selectedSuiteInfo?.name}</strong>
                </span>
              </div>

              <div className="flex items-center gap-2">
                {isDone && (
                  <button
                    onClick={handleDownloadMarkdown}
                    className="flex items-center gap-1.5 px-3.5 py-2 rounded-xl border border-border bg-card hover:bg-muted text-xs font-medium text-foreground transition-all shadow-sm"
                  >
                    <Download className="w-3.5 h-3.5 text-primary" />
                    <span>دانلود گزارش منتور (MD)</span>
                  </button>
                )}

                <button
                  onClick={handleStartRun}
                  disabled={isRunning || isStarting}
                  className="flex items-center gap-2 px-5 py-2 rounded-xl bg-primary text-primary-foreground font-semibold text-xs hover:opacity-95 active:scale-[0.98] transition-all disabled:opacity-50 shadow-md shadow-primary/20"
                >
                  {isRunning ? (
                    <>
                      <RotateCcw className="w-3.5 h-3.5 animate-spin" />
                      <span>درحال اجرای بنچ‌مارک...</span>
                    </>
                  ) : (
                    <>
                      <Play className="w-3.5 h-3.5 fill-current" />
                      <span>اجرای ارزیابی (Start Benchmark Run)</span>
                    </>
                  )}
                </button>
              </div>
            </div>
          </div>

          {/* Live Progress Bar */}
          {isRunning && (
            <div className="bg-card border border-primary/30 rounded-2xl p-4 shadow-lg animate-in fade-in duration-200">
              <div className="flex items-center justify-between text-xs mb-2">
                <span className="font-semibold text-primary flex items-center gap-2">
                  <span className="w-2.5 h-2.5 rounded-full bg-primary animate-ping" />
                  پیشرفت پردازش بنچ‌مارک: {runStatus?.progress_percentage.toFixed(0)}%
                </span>
                <span className="font-mono text-muted-foreground">
                  {runStatus?.processed_items} / {runStatus?.total_items} مورد •{" "}
                  {runStatus?.elapsed_seconds.toFixed(1)}s
                </span>
              </div>
              <div className="w-full h-2.5 bg-muted rounded-full overflow-hidden border border-border/50">
                <div
                  className="h-full bg-gradient-to-r from-primary to-indigo-500 rounded-full transition-all duration-300"
                  style={{ width: `${runStatus?.progress_percentage || 5}%` }}
                />
              </div>
              {runStatus?.current_item_query && (
                <div className="mt-2.5 text-[11px] text-muted-foreground truncate font-sans">
                  <span className="font-medium text-foreground">پرسمان جاری: </span>
                  {runStatus.current_item_query}
                </div>
              )}
            </div>
          )}

          {/* Error Banner */}
          {error && (
            <div className="bg-destructive/10 border border-destructive/30 rounded-2xl p-4 text-xs text-destructive flex items-center gap-2">
              <AlertTriangle className="w-4 h-4 shrink-0" />
              <span>{error}</span>
            </div>
          )}

          {/* Results Area */}
          {report && (
            <div className="space-y-4">
              {/* Tab Selector */}
              <div className="flex items-center justify-between border-b border-border/80 pb-2">
                <div className="flex items-center gap-2">
                  <button
                    onClick={() => setActiveTab("scorecard")}
                    className={`px-3 py-1.5 rounded-xl text-xs font-semibold transition-all ${
                      activeTab === "scorecard"
                        ? "bg-primary/15 text-primary border border-primary/30"
                        : "hover:bg-muted text-muted-foreground"
                    }`}
                  >
                    شاخص‌های کلیدی (KPI Scorecard)
                  </button>
                  <button
                    onClick={() => setActiveTab("matrix")}
                    className={`px-3 py-1.5 rounded-xl text-xs font-semibold transition-all ${
                      activeTab === "matrix"
                        ? "bg-primary/15 text-primary border border-primary/30"
                        : "hover:bg-muted text-muted-foreground"
                    }`}
                  >
                    ماتریس ۲×۲ تشخیصی (Confusion Matrix)
                  </button>
                  <button
                    onClick={() => setActiveTab("failures")}
                    className={`px-3 py-1.5 rounded-xl text-xs font-semibold transition-all ${
                      activeTab === "failures"
                        ? "bg-primary/15 text-primary border border-primary/30"
                        : "hover:bg-muted text-muted-foreground"
                    }`}
                  >
                    تحلیل خطاهای ریشه‌ای ({report.failing_cases.length})
                  </button>
                </div>

                <div className="text-[11px] text-muted-foreground font-mono">
                  Report ID: <span className="font-semibold text-foreground">{report.report_id}</span>
                </div>
              </div>

              {/* Tab 1: KPI Scorecard */}
              {activeTab === "scorecard" && (
                <div className="space-y-4">
                  {/* Big Headline Cards */}
                  <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                    <div className="bg-card border border-border/80 rounded-2xl p-3.5 text-center">
                      <div className="text-[11px] text-muted-foreground font-medium mb-1">
                        نرخ قبولی کل (Pass Rate)
                      </div>
                      <div className="text-2xl font-black text-foreground">
                        {report.overall_pass_rate.toFixed(1)}%
                      </div>
                      <div className="text-[10px] mt-1 font-mono text-muted-foreground">
                        بیس‌لاین: ۶۲.۷٪ • هدف: ۸۸.۰٪
                      </div>
                    </div>

                    <div className="bg-card border border-border/80 rounded-2xl p-3.5 text-center">
                      <div className="text-[11px] text-muted-foreground font-medium mb-1">
                        دقت بازیابی (Hit@5)
                      </div>
                      <div className="text-2xl font-black text-foreground">
                        {(
                          (Object.values(report.suite_summaries).reduce(
                            (acc, s) => acc + s.hit_at_5 * s.total_items,
                            0
                          ) /
                            (report.total_items || 1)) *
                          100
                        ).toFixed(1)}
                        %
                      </div>
                      <div className="text-[10px] mt-1 font-mono text-muted-foreground">
                        بیس‌لاین: ۶۸.۴٪ • هدف: ۹۲.۰٪
                      </div>
                    </div>

                    <div className="bg-card border border-border/80 rounded-2xl p-3.5 text-center">
                      <div className="text-[11px] text-muted-foreground font-medium mb-1">
                        درستی سنتز (Faithfulness)
                      </div>
                      <div className="text-2xl font-black text-foreground">
                        {(
                          (Object.values(report.suite_summaries).reduce(
                            (acc, s) => acc + s.faithfulness * s.total_items,
                            0
                          ) /
                            (report.total_items || 1)) *
                          100
                        ).toFixed(1)}
                        %
                      </div>
                      <div className="text-[10px] mt-1 font-mono text-muted-foreground">
                        بیس‌لاین: ۷۱.۰٪ • هدف: ۹۰.۰٪
                      </div>
                    </div>

                    <div className="bg-card border border-border/80 rounded-2xl p-3.5 text-center">
                      <div className="text-[11px] text-muted-foreground font-medium mb-1">
                        مهار توهم (Catch Rate)
                      </div>
                      <div className="text-2xl font-black text-emerald-500">
                        {(report.confusion_matrix.hallucination_catch_rate
                          ? report.confusion_matrix.hallucination_catch_rate * 100
                          : 100
                        ).toFixed(1)}
                        %
                      </div>
                      <div className="text-[10px] mt-1 font-mono text-muted-foreground">
                        TN / (TN + FP) • هدف: ۹۵.۰٪
                      </div>
                    </div>
                  </div>

                  {/* Suite Breakdown Table */}
                  <div className="border border-border/80 rounded-2xl overflow-hidden bg-card">
                    <div className="p-3 border-b border-border/70 text-xs font-semibold bg-muted/20">
                      عملکرد تفکیکی سوئیت‌ها (Per-Suite Calibrated Breakdown)
                    </div>
                    <div className="overflow-x-auto">
                      <table className="w-full text-xs text-right">
                        <thead className="bg-muted/40 text-muted-foreground font-medium border-b border-border/60">
                          <tr>
                            <th className="p-2.5">سوئیت بنچ‌مارک</th>
                            <th className="p-2.5 text-center">تعداد</th>
                            <th className="p-2.5 text-center">قبول</th>
                            <th className="p-2.5 text-center">نرخ قبولی</th>
                            <th className="p-2.5 text-center">Hit@1</th>
                            <th className="p-2.5 text-center">Hit@5</th>
                            <th className="p-2.5 text-center">MRR</th>
                            <th className="p-2.5 text-center">Faithfulness</th>
                            <th className="p-2.5 text-center">میانگین تاخیر</th>
                          </tr>
                        </thead>
                        <tbody className="divide-y divide-border/50">
                          {Object.entries(report.suite_summaries).map(([sKey, summary]) => (
                            <tr key={sKey} className="hover:bg-muted/20 transition-colors">
                              <td className="p-2.5 font-semibold text-foreground">{summary.suite}</td>
                              <td className="p-2.5 text-center font-mono">{summary.total_items}</td>
                              <td className="p-2.5 text-center font-mono text-emerald-500 font-bold">
                                {summary.passed_items}
                              </td>
                              <td className="p-2.5 text-center font-mono font-bold text-foreground">
                                {summary.pass_rate.toFixed(1)}%
                              </td>
                              <td className="p-2.5 text-center font-mono">
                                {(summary.hit_at_1 * 100).toFixed(1)}%
                              </td>
                              <td className="p-2.5 text-center font-mono">
                                {(summary.hit_at_5 * 100).toFixed(1)}%
                              </td>
                              <td className="p-2.5 text-center font-mono">{summary.mrr.toFixed(3)}</td>
                              <td className="p-2.5 text-center font-mono">
                                {(summary.faithfulness * 100).toFixed(1)}%
                              </td>
                              <td className="p-2.5 text-center font-mono">
                                {summary.mean_latency_seconds.toFixed(2)}s
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                </div>
              )}

              {/* Tab 2: 2x2 Decoupled Confusion Matrix */}
              {activeTab === "matrix" && (
                <div className="space-y-4">
                  <div className="bg-muted/20 border border-border/80 rounded-2xl p-4">
                    <div className="text-xs font-bold text-foreground mb-1">
                      ماتریس تفکیک خطای بازیابی از توهم سنتز (Decoupled Attribution Grid)
                    </div>
                    <p className="text-[11px] text-muted-foreground mb-4">
                      پاسخ به چالش اساسی منتور: آیا خطا ناشی از نیافتن متن مرجع بوده یا مدل زبانی دچار
                      توهم شده است؟
                    </p>

                    <div className="grid grid-cols-2 gap-3 max-w-2xl mx-auto">
                      {/* Top-Left: True Positive */}
                      <div className="p-4 rounded-xl border border-emerald-500/40 bg-emerald-500/10 flex flex-col justify-between">
                        <div>
                          <div className="flex items-center justify-between text-xs font-bold text-emerald-500">
                            <span>True Positive (TP)</span>
                            <span className="text-lg font-black">
                              {report.confusion_matrix.true_positive}
                            </span>
                          </div>
                          <p className="text-[11px] text-emerald-600/90 dark:text-emerald-400/90 mt-1">
                            موفقیت کامل در بازیابی سند و پاسخ وفادارانه مدل.
                          </p>
                        </div>
                        <div className="text-[10px] font-mono text-muted-foreground mt-3 pt-2 border-t border-emerald-500/20">
                          Retrieval Hit = 1 • Faithfulness = 1
                        </div>
                      </div>

                      {/* Top-Right: True Negative */}
                      <div className="p-4 rounded-xl border border-emerald-500/40 bg-emerald-500/10 flex flex-col justify-between">
                        <div>
                          <div className="flex items-center justify-between text-xs font-bold text-emerald-500">
                            <span>True Negative (TN)</span>
                            <span className="text-lg font-black">
                              {report.confusion_matrix.true_negative}
                            </span>
                          </div>
                          <p className="text-[11px] text-emerald-600/90 dark:text-emerald-400/90 mt-1">
                            مهار امنیتی: عدم‌پاسخ تمیز به پرسمان‌های امنیتی یا غیرقابل پاسخ.
                          </p>
                        </div>
                        <div className="text-[10px] font-mono text-muted-foreground mt-3 pt-2 border-t border-emerald-500/20">
                          Safe Persian Refusal • 0 Hallucination
                        </div>
                      </div>

                      {/* Bottom-Left: False Positive */}
                      <div className="p-4 rounded-xl border border-rose-500/40 bg-rose-500/10 flex flex-col justify-between">
                        <div>
                          <div className="flex items-center justify-between text-xs font-bold text-rose-500">
                            <span>False Positive (FP)</span>
                            <span className="text-lg font-black">
                              {report.confusion_matrix.false_positive}
                            </span>
                          </div>
                          <p className="text-[11px] text-rose-600/90 dark:text-rose-400/90 mt-1">
                            خطای مدل زبانی: سند بازیابی شد اما مدل دچار توهم یا انحراف گردید.
                          </p>
                        </div>
                        <div className="text-[10px] font-mono text-muted-foreground mt-3 pt-2 border-t border-rose-500/20">
                          Retrieval Hit = 1 • Faithfulness = 0
                        </div>
                      </div>

                      {/* Bottom-Right: False Negative */}
                      <div className="p-4 rounded-xl border border-amber-500/40 bg-amber-500/10 flex flex-col justify-between">
                        <div>
                          <div className="flex items-center justify-between text-xs font-bold text-amber-500">
                            <span>False Negative (FN)</span>
                            <span className="text-lg font-black">
                              {report.confusion_matrix.false_negative}
                            </span>
                          </div>
                          <p className="text-[11px] text-amber-600/90 dark:text-amber-400/90 mt-1">
                            خطای بازیابی: سند لازم در پایگاه پیدا نشد و مدل بدون سند پاسخ داد.
                          </p>
                        </div>
                        <div className="text-[10px] font-mono text-muted-foreground mt-3 pt-2 border-t border-amber-500/20">
                          Retrieval Hit = 0 • Attempted Answer
                        </div>
                      </div>
                    </div>

                    <div className="mt-4 p-3 bg-card border border-border/60 rounded-xl text-xs space-y-1.5 font-mono">
                      <div className="flex items-center justify-between text-muted-foreground">
                        <span>نرخ مهار توهم (Hallucination Catch Rate):</span>
                        <strong className="text-foreground">
                          TN / (TN + FP) ={" "}
                          {(
                            (report.confusion_matrix.hallucination_catch_rate || 1.0) * 100
                          ).toFixed(1)}
                          %
                        </strong>
                      </div>
                      <div className="flex items-center justify-between text-muted-foreground">
                        <span>پوشش بازیابی (Retrieval Coverage):</span>
                        <strong className="text-foreground">
                          (TP + FP) / Total ={" "}
                          {(((report.confusion_matrix.true_positive +
                            report.confusion_matrix.false_positive) /
                            (report.total_items || 1)) *
                            100).toFixed(1)}
                          %
                        </strong>
                      </div>
                    </div>
                  </div>
                </div>
              )}

              {/* Tab 3: Failures Explorer */}
              {activeTab === "failures" && (
                <div className="space-y-3">
                  {report.failing_cases.length === 0 ? (
                    <div className="p-6 text-center border border-border/80 rounded-2xl bg-card">
                      <CheckCircle2 className="w-8 h-8 text-emerald-500 mx-auto mb-2" />
                      <div className="text-sm font-bold text-foreground">
                        تبریک! هیچ خطایی در این اجرا ثبت نشد.
                      </div>
                      <p className="text-xs text-muted-foreground mt-1">
                        تمام موارد آزمون با موفقیت سپری شدند.
                      </p>
                    </div>
                  ) : (
                    report.failing_cases.map((fail, idx) => (
                      <div
                        key={idx}
                        className="bg-card border border-border/80 rounded-2xl p-4 space-y-2 text-xs"
                      >
                        <div className="flex items-center justify-between">
                          <div className="flex items-center gap-2">
                            <span className="font-bold text-foreground font-mono">
                              {fail.item_id}
                            </span>
                            <span className="bg-muted px-2 py-0.5 rounded text-[10px] text-muted-foreground">
                              {fail.suite}
                            </span>
                          </div>
                          <a
                            href={fail.trace_url}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="flex items-center gap-1 text-[11px] text-primary hover:underline"
                          >
                            <span>مشاهده ردپا در Langfuse</span>
                            <ExternalLink className="w-3 h-3" />
                          </a>
                        </div>

                        <div>
                          <span className="text-muted-foreground font-medium">پرسمان: </span>
                          <span className="text-foreground font-semibold">{fail.query}</span>
                        </div>

                        <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 p-2.5 rounded-xl bg-muted/30 border border-border/50 text-[11px]">
                          <div>
                            <span className="text-muted-foreground font-medium block mb-0.5">
                              پاسخ مورد انتظار (Gold):
                            </span>
                            <span className="text-foreground font-mono">{fail.expected}</span>
                          </div>
                          <div>
                            <span className="text-muted-foreground font-medium block mb-0.5">
                              پاسخ مدل (Actual):
                            </span>
                            <span className="text-foreground font-mono">{fail.actual}</span>
                          </div>
                        </div>

                        <div className="text-[11px] text-amber-500 flex items-center gap-1.5 pt-1">
                          <AlertTriangle className="w-3.5 h-3.5 shrink-0" />
                          <span>علت خطا: {fail.reason}</span>
                        </div>
                      </div>
                    ))
                  )}
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

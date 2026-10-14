"use client"

import React, { useState, useEffect } from "react"
import { OperationalMetrics } from "@/types"
import { X, RefreshCw, Activity, Zap, ShieldAlert, Database, Clock, DollarSign } from "lucide-react"

interface MetricsModalProps {
  isOpen: boolean
  onClose: () => void
}

export function MetricsModal({ isOpen, onClose }: MetricsModalProps) {
  const [metrics, setMetrics] = useState<OperationalMetrics | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const fetchMetrics = async () => {
    setIsLoading(true)
    setError(null)
    try {
      const res = await fetch("/api/v1/metrics")
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      const data = await res.json()
      setMetrics(data)
    } catch (e: any) {
      setError(e.message || "Failed to load metrics")
    } finally {
      setIsLoading(false)
    }
  }

  useEffect(() => {
    if (isOpen) {
      fetchMetrics()
      const interval = setInterval(fetchMetrics, 3000)
      return () => clearInterval(interval)
    }
  }, [isOpen])

  if (!isOpen) return null

  return (
    <div className="fixed inset-0 bg-background/80 backdrop-blur-sm z-50 flex items-center justify-center p-4">
      <div className="bg-card border border-border rounded-2xl max-w-xl w-full flex flex-col shadow-2xl animate-in zoom-in-95 duration-150">
        {/* Header */}
        <div className="p-4 border-b border-border/80 flex items-center justify-between">
          <div className="flex items-center gap-2.5">
            <div className="w-8 h-8 rounded-xl bg-primary/10 text-primary flex items-center justify-center">
              <Activity className="w-4 h-4" />
            </div>
            <div>
              <h2 className="text-sm font-bold flex items-center gap-2">
                <span>تله‌متری زنده و شاخص‌های سلامت سامانه</span>
                <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
              </h2>
              <p className="text-[11px] text-muted-foreground font-mono">
                Production KPIs (Concurrency, Latency & Fallbacks)
              </p>
            </div>
          </div>
          <div className="flex items-center gap-1.5">
            <button
              onClick={fetchMetrics}
              className={`p-1.5 hover:bg-muted text-muted-foreground hover:text-foreground rounded-lg transition-colors ${
                isLoading ? "animate-spin" : ""
              }`}
              title="بروزرسانی"
            >
              <RefreshCw className="w-4 h-4" />
            </button>
            <button
              onClick={onClose}
              className="p-1.5 hover:bg-muted text-muted-foreground hover:text-foreground rounded-lg transition-colors"
            >
              <X className="w-4 h-4" />
            </button>
          </div>
        </div>

        {/* Body */}
        <div className="p-4 space-y-4 overflow-y-auto max-h-[70vh]">
          {error && (
            <div className="p-3 rounded-xl bg-destructive/10 text-destructive text-xs border border-destructive/20">
              خطا در دریافت تله‌متری: {error}
            </div>
          )}

          {/* Metric Tiles Grid */}
          <div className="grid grid-cols-2 sm:grid-cols-3 gap-2.5">
            {/* Active Requests */}
            <div className="p-3 rounded-xl bg-muted/40 border border-border/70 space-y-1">
              <div className="text-[10px] text-muted-foreground flex items-center gap-1">
                <Activity className="w-3 h-3 text-primary" />
                <span>درخواست‌های همزمان</span>
              </div>
              <div className="text-lg font-bold font-mono">
                {metrics ? `${metrics.active_requests} / 50` : "-"}
              </div>
              <div className="text-[9px] text-muted-foreground">ظرفیت کل: ۵۰ کاربر</div>
            </div>

            {/* Semaphore Queue Wait */}
            <div className="p-3 rounded-xl bg-muted/40 border border-border/70 space-y-1">
              <div className="text-[10px] text-muted-foreground flex items-center gap-1">
                <Clock className="w-3 h-3 text-amber-500" />
                <span>تاخیر صف همزمانی</span>
              </div>
              <div className="text-lg font-bold font-mono text-amber-500">
                {metrics ? `${metrics.avg_queue_wait_ms} ms` : "-"}
              </div>
              <div className="text-[9px] text-muted-foreground">هدف SLA: زیر ۲۰۰ms</div>
            </div>

            {/* Circuit Breaker Trip Rate */}
            <div className="p-3 rounded-xl bg-muted/40 border border-border/70 space-y-1">
              <div className="text-[10px] text-muted-foreground flex items-center gap-1">
                <ShieldAlert className="w-3 h-3 text-emerald-500" />
                <span>شکست مدار Reranker</span>
              </div>
              <div className="text-lg font-bold font-mono text-emerald-500">
                {metrics ? `${(metrics.circuit_breaker_trip_rate * 100).toFixed(1)}%` : "-"}
              </div>
              <div className="text-[9px] text-muted-foreground">هدف سلامت: زیر ۲.۰%</div>
            </div>

            {/* Cache Hit Rate */}
            <div className="p-3 rounded-xl bg-muted/40 border border-border/70 space-y-1">
              <div className="text-[10px] text-muted-foreground flex items-center gap-1">
                <Database className="w-3 h-3 text-blue-500" />
                <span>نرخ کش Redis</span>
              </div>
              <div className="text-lg font-bold font-mono text-blue-500">
                {metrics ? `${(metrics.cache_hit_rate * 100).toFixed(1)}%` : "-"}
              </div>
              <div className="text-[9px] text-muted-foreground">پاسخ در &lt; ۱۵ms</div>
            </div>

            {/* p95 Latency */}
            <div className="p-3 rounded-xl bg-muted/40 border border-border/70 space-y-1">
              <div className="text-[10px] text-muted-foreground flex items-center gap-1">
                <Zap className="w-3 h-3 text-purple-500" />
                <span>تاخیر صدک ۹۵ (p95)</span>
              </div>
              <div className="text-lg font-bold font-mono">
                {metrics ? `${(metrics.latency_p95_ms / 1000).toFixed(2)} s` : "-"}
              </div>
              <div className="text-[9px] text-muted-foreground">هدف: زیر ۲.۵s</div>
            </div>

            {/* Cost per 1k */}
            <div className="p-3 rounded-xl bg-muted/40 border border-border/70 space-y-1">
              <div className="text-[10px] text-muted-foreground flex items-center gap-1">
                <DollarSign className="w-3 h-3 text-emerald-500" />
                <span>هزینه ۱۰۰۰ پرسش (۱k Q)</span>
              </div>
              <div className="text-sm md:text-base font-bold font-mono text-foreground">
                {metrics
                  ? `$${(metrics.est_cost_per_1k_min_usd ?? metrics.est_cost_per_1k_usd).toFixed(2)} – $${(metrics.est_cost_per_1k_max_usd ?? 6.0).toFixed(2)}`
                  : "-"}
              </div>
              <div className="text-[9px] text-muted-foreground">بازه مدل اقتصادی تا پیشرفته</div>
            </div>
          </div>

          {/* Capacity Planning & Budgeting Section */}
          <div className="p-3.5 rounded-xl bg-muted/20 border border-border/60 text-xs space-y-2.5">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-1.5 font-bold text-foreground">
                <DollarSign className="w-3.5 h-3.5 text-emerald-500" />
                <span>برنامه‌ریزی ظرفیت و بودجه (Capacity Planning - Cost / 1k Q)</span>
              </div>
              <span className="text-[10px] text-muted-foreground">
                دید پیش‌بینی هزینه برای ۱۰۰۰ دانش‌آموز در روز
              </span>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-2.5">
              {/* Min Tier */}
              <div className="p-2.5 rounded-lg bg-background/70 border border-emerald-500/20 space-y-1">
                <div className="flex justify-between items-center">
                  <span className="font-semibold text-foreground text-[11px]">مدل اقتصادی (Min Tier):</span>
                  <span className="text-emerald-500 font-bold font-mono text-sm">
                    ${metrics ? (metrics.est_cost_per_1k_min_usd ?? metrics.est_cost_per_1k_usd).toFixed(2) : "0.55"} / 1k Q
                  </span>
                </div>
                <div className="text-[10px] text-muted-foreground space-y-0.5 pt-0.5 border-t border-border/30">
                  <div className="flex justify-between">
                    <span>ورودی (Cache Miss):</span>
                    <span className="font-mono text-foreground">$0.15 / 1M</span>
                  </div>
                  <div className="flex justify-between">
                    <span>ورودی (Cache Hit):</span>
                    <span className="font-mono text-foreground">$0.003 / 1M</span>
                  </div>
                  <div className="flex justify-between">
                    <span>خروجی (Completion):</span>
                    <span className="font-mono text-foreground">$0.60 / 1M</span>
                  </div>
                </div>
              </div>

              {/* Max Tier */}
              <div className="p-2.5 rounded-lg bg-background/70 border border-purple-500/20 space-y-1">
                <div className="flex justify-between items-center">
                  <span className="font-semibold text-foreground text-[11px]">مدل پرچمدار (Max Tier):</span>
                  <span className="text-purple-400 font-bold font-mono text-sm">
                    ${metrics ? (metrics.est_cost_per_1k_max_usd ?? 6.00).toFixed(2) : "6.00"} / 1k Q
                  </span>
                </div>
                <div className="text-[10px] text-muted-foreground space-y-0.5 pt-0.5 border-t border-border/30">
                  <div className="flex justify-between">
                    <span>ورودی (Input):</span>
                    <span className="font-mono text-foreground">$1.50 / 1M</span>
                  </div>
                  <div className="flex justify-between">
                    <span>کش ورودی (Cache Hit):</span>
                    <span className="font-mono text-emerald-400 font-semibold">رایگان (Free)</span>
                  </div>
                  <div className="flex justify-between">
                    <span>خروجی (Completion):</span>
                    <span className="font-mono text-foreground">$7.50 / 1M</span>
                  </div>
                </div>
              </div>
            </div>

            <div className="text-[10px] text-muted-foreground bg-muted/30 p-2 rounded-lg leading-relaxed border border-border/30">
              💡 <strong>راهنمای مدیر محصول:</strong> نمایش مجموع کل هزینه ($0.0007) دید درستی از هزینه‌های مقیاس به مدیر محصول نمی‌دهد. اما شاخص <strong>Cost / 1k Q</strong> نشان می‌دهد در صورت استفاده روزانه ۱۰۰۰ دانش‌آموز، هزینه بر اساس مدل انتخابی بین <strong>${metrics ? (metrics.est_cost_per_1k_min_usd ?? 0.55).toFixed(2) : "0.55"}</strong> تا <strong>${metrics ? (metrics.est_cost_per_1k_max_usd ?? 6.00).toFixed(2) : "6.00"}</strong> دلار خواهد بود.
            </div>
          </div>

          {/* Latency Distribution Table */}
          <div className="p-3 rounded-xl bg-muted/20 border border-border/60 text-xs space-y-2">
            <div className="font-semibold text-[11px] text-muted-foreground">
              توزیع زمانی تاخیر پاسخ‌ها (Latency Distribution)
            </div>
            <div className="grid grid-cols-3 gap-2 text-center font-mono text-[11px]">
              <div className="p-2 bg-background/60 rounded-lg border border-border/40">
                <div className="text-muted-foreground text-[10px]">p50 (میانه)</div>
                <div className="font-bold text-foreground mt-0.5">
                  {metrics ? `${metrics.latency_p50_ms} ms` : "-"}
                </div>
              </div>
              <div className="p-2 bg-background/60 rounded-lg border border-border/40">
                <div className="text-muted-foreground text-[10px]">p95</div>
                <div className="font-bold text-foreground mt-0.5">
                  {metrics ? `${metrics.latency_p95_ms} ms` : "-"}
                </div>
              </div>
              <div className="p-2 bg-background/60 rounded-lg border border-border/40">
                <div className="text-muted-foreground text-[10px]">p99 (بدترین حالت)</div>
                <div className="font-bold text-foreground mt-0.5">
                  {metrics ? `${metrics.latency_p99_ms} ms` : "-"}
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Footer */}
        <div className="p-3 border-t border-border/80 flex items-center justify-between text-[11px] text-muted-foreground bg-muted/20">
          <span>داده‌ها هر ۳ ثانیه به صورت خودکار بروزرسانی می‌شوند.</span>
          <button
            onClick={onClose}
            className="px-4 py-1.5 bg-primary text-primary-foreground font-semibold rounded-lg hover:opacity-95 transition-all text-xs"
          >
            بستن
          </button>
        </div>
      </div>
    </div>
  )
}

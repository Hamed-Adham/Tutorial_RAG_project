export interface Citation {
  doc_id: string
  filename: string
  page: number
  chunk_id?: string
  score?: number
  metadata?: {
    title?: string
  }
}

export interface LatencyProfile {
  queue_wait_ms?: number
  rewrite_ms?: number
  retrieval_ms?: number
  rerank_ms?: number
  expansion_ms?: number
  synthesis_ms?: number
  total_ms?: number
  [key: string]: number | undefined
}

export interface Cost1kEstimate {
  min_1k_usd: number
  max_1k_usd: number
  cost_single_min_usd?: number
  cost_single_max_usd?: number
  prompt_tokens?: number
  completion_tokens?: number
}

export interface Message {
  id: string
  turn_id?: string
  role: "user" | "assistant"
  content: string
  timestamp: string
  citations?: Citation[]
  latency_profile?: LatencyProfile
  is_refusal?: boolean
  effective_query?: string
  query_rewritten?: boolean
  trace_id?: string
  status_title?: string
  cost_1k?: Cost1kEstimate
  conversational_context_badge?: string
}

export interface SelectedContextTurn {
  id: string
  content: string
  role: "user" | "assistant"
}

export interface Session {
  id: string
  title: string
  createdAt: string
  updatedAt: string
  selectedChapters?: number[]
}

export interface Chapter {
  id: number
  title: string
  subtitle: string
  topics: string[]
}

export interface SystemHealth {
  status: "healthy" | "degraded" | "unhealthy"
  qdrant_connected: boolean
  bm25_loaded: boolean
  redis_connected: boolean
  cross_encoder_warm: boolean
  uptime_seconds: number
}

export interface OperationalMetrics {
  active_requests: number
  available_semaphore_slots: number
  avg_queue_wait_ms: number
  total_requests: number
  circuit_breaker_trips: number
  circuit_breaker_trip_rate: number
  cache_hits: number
  cache_hit_rate: number
  latency_p50_ms: number
  latency_p95_ms: number
  latency_p99_ms: number
  est_cost_per_1k_usd: number
  est_cost_per_1k_min_usd?: number
  est_cost_per_1k_max_usd?: number
}

export type EvaluationScope = "end_to_end" | "retrieve_only" | "synthesis_only" | "gates_only"
export type SplitType = "dev" | "test"
export type SuiteType =
  | "science_curriculum"
  | "administrative_faq"
  | "conversational_memory"
  | "subquery_decomposition"
  | "adversarial_refusal"

export interface SuiteInfo {
  id: string
  name: string
  description: string
  total_items: number
  dev_items: number
  test_items: number
  difficulty_profile: string
}

export interface ConfusionMatrixResult {
  true_positive: number
  false_positive: number
  false_negative: number
  true_negative: number
  total?: number
  hallucination_catch_rate?: number
  retrieval_coverage?: number
}

export interface SuiteSummary {
  suite: string
  total_items: number
  passed_items: number
  pass_rate: number
  hit_at_1: number
  hit_at_5: number
  mrr: number
  faithfulness: number
  answer_relevancy: number
  mean_latency_seconds: number
  total_tokens: number
}

export interface FailingCase {
  item_id: string
  suite: string
  query: string
  expected: string
  actual: string
  reason: string
  trace_url: string
}

export interface EvaluationReport {
  report_id: string
  timestamp: string
  split_evaluated: string
  judge_backend: string
  total_items: number
  overall_pass_rate: number
  baseline_pass_rate: number
  score_delta: number
  suite_summaries: Record<string, SuiteSummary>
  confusion_matrix: ConfusionMatrixResult
  mean_latency_seconds: number
  p95_latency_seconds: number
  total_cost_usd: number
  failing_cases: FailingCase[]
}

export interface EvaluationRunStatus {
  run_id: string
  status: "queued" | "running" | "completed" | "failed"
  progress_percentage: number
  total_items: number
  processed_items: number
  current_item_query?: string
  elapsed_seconds: number
  report?: EvaluationReport
  error?: string
}

export interface EvaluationRunRequest {
  suite?: string | null
  split: SplitType
  scope: EvaluationScope
  judge_model: string
  limit?: number
}


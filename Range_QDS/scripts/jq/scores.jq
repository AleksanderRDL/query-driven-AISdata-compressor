def delta($a; $b):
  if ($a | type) == "number" and ($b | type) == "number" then $a - $b else null end;

{
  mlqds: .matched.MLQDS.query_local_utility_score,
  uniform: .matched.uniform.query_local_utility_score,
  douglas_peucker: .matched.DouglasPeucker.query_local_utility_score,
  mlqds_vs_uniform: delta(.matched.MLQDS.query_local_utility_score; .matched.uniform.query_local_utility_score),
  mlqds_vs_douglas_peucker: delta(.matched.MLQDS.query_local_utility_score; .matched.DouglasPeucker.query_local_utility_score),
  beats_uniform: (
    (.matched.MLQDS.query_local_utility_score | type) == "number"
    and (.matched.uniform.query_local_utility_score | type) == "number"
    and .matched.MLQDS.query_local_utility_score > .matched.uniform.query_local_utility_score
  ),
  beats_douglas_peucker: (
    (.matched.MLQDS.query_local_utility_score | type) == "number"
    and (.matched.DouglasPeucker.query_local_utility_score | type) == "number"
    and .matched.MLQDS.query_local_utility_score > .matched.DouglasPeucker.query_local_utility_score
  ),
  final_metrics_mode,
  final_claim_status: .final_claim_summary.status
}

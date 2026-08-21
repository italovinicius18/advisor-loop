from advisor_loop.episode import Episode, Outcome, Recommendation


class ProposalBackend:
    """Backend de modo-proposta: nao existe oracle de referencia. O advisor
    propoe uma configuracao e ela e apenas registrada - a avaliacao real
    (grading) acontece depois via execucao real no cluster, fora deste escopo."""

    def execute(self, ep: Episode, rec: Recommendation) -> Outcome:
        return Outcome(status="proposed", per_query={}, wall_clock_s=0.0, cost_usd=0.0)

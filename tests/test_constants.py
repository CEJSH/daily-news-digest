from daily_news_digest.core.constants import DEDUPE_CLUSTER_DOMAINS


def test_energy_domain_has_natural_gas_and_lng() -> None:
    energy = DEDUPE_CLUSTER_DOMAINS.get("에너지", set())
    assert "natural gas" in energy
    assert "lng" in energy

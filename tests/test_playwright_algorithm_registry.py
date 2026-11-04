"""The page lists an algorithm that was registered after its markup was written."""

from grn_sim.inference import registry
from tests.test_algorithm_registry import _EveryOtherEdge


def test_page_adds_a_late_registered_algorithm(app_url, browser):
    registry.register(registry.AlgorithmSpec(key="late_algo", label="Late Algo (test)", build=lambda c: _EveryOtherEdge(),
                                             color="#123456", order=6000))
    try:
        page = browser.new_page()
        page.goto(app_url, wait_until="networkidle")
        page.wait_for_selector(".alg-cb[value='late_algo']", state="attached", timeout=15000)
        page.check(".alg-cb[value='late_algo']")                 # it can be ticked (the page restores the open run's selection first)
        assert page.locator(".alg-cb[value='late_algo']").is_checked()
        assert "Late Algo (test)" in page.locator("label:has(.alg-cb[value='late_algo'])").inner_text()
        # the algorithm menus and the chart colour map know it too
        assert page.evaluate("[...document.querySelectorAll('#select-infer-alg option')].some(o => o.value === 'late_algo')")
        assert page.evaluate("ALG_UI['late_algo'][1]") == "#123456"
        # the count labels follow
        total = page.evaluate("new Set([...document.querySelectorAll('.alg-cb')].map(c => c.value)).size")
        assert page.locator(".alg-total").first.inner_text() == str(total)
        page.close()
    finally:
        registry.unregister("late_algo")

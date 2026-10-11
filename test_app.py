"""Direct tests for FastAPI route handlers without extra HTTP dependencies."""

import unittest
from fastapi import HTTPException
from app import (
    RunRequest,
    CompareRequest,
    run_single,
    compare_all,
    get_algorithms_info,
)


class TestAppHandlers(unittest.TestCase):
    def test_info_endpoint(self):
        data = get_algorithms_info()
        self.assertIn("BFS", data)
        self.assertIn("DFS", data)
        self.assertIn("Dijkstra", data)
        self.assertIn("A*", data)
        self.assertTrue(data["A*"]["guarantees_shortest_path"])

    def test_run_single_success(self):
        req = RunRequest(
            rows=15,
            cols=15,
            start=[0, 0],
            end=[14, 14],
            walls=[[1, 0], [1, 1], [1, 2]],
            algorithm="A*",
        )
        data = run_single(req)
        self.assertTrue(data["success"])
        self.assertGreater(len(data["explored"]), 0)
        self.assertGreater(data["path_length"], 0)
        self.assertEqual(data["path"][0], [0, 0])
        self.assertEqual(data["path"][-1], [14, 14])

    def test_compare_all_endpoint(self):
        req = CompareRequest(
            rows=10,
            cols=10,
            start=[0, 0],
            end=[9, 9],
            walls=[],
        )
        data = compare_all(req)
        self.assertIn("results", data)
        self.assertIn("summary", data)
        self.assertEqual(len(data["summary"]), 4)

        # BFS, Dijkstra, and A* all give optimal 18 steps
        results = data["results"]
        self.assertEqual(results["BFS"]["path_length"], 18)
        self.assertEqual(results["Dijkstra"]["path_length"], 18)
        self.assertEqual(results["A*"]["path_length"], 18)

    def test_unreachable_target(self):
        req = RunRequest(
            rows=10,
            cols=10,
            start=[0, 0],
            end=[9, 9],
            walls=[[8, 9], [9, 8]],
            algorithm="BFS",
        )
        data = run_single(req)
        self.assertFalse(data["success"])
        self.assertEqual(data["path"], [])
        self.assertIn("unreachable", data["message"].lower())

    def test_bad_request_out_of_bounds(self):
        with self.assertRaises(HTTPException) as ctx:
            req = RunRequest(
                rows=10,
                cols=10,
                start=[-1, 0],
                end=[9, 9],
                algorithm="A*",
            )
            run_single(req)
        self.assertEqual(ctx.exception.status_code, 400)


if __name__ == "__main__":
    unittest.main()

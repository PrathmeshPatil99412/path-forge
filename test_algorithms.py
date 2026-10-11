"""Comprehensive tests for PathForge algorithms."""

import unittest
from algorithms import (
    bfs,
    dfs,
    dijkstra,
    astar,
    run_algorithm,
    run_all_algorithms,
    manhattan_distance,
)


class TestAlgorithms(unittest.TestCase):
    def setUp(self):
        self.rows = 10
        self.cols = 10
        self.start = (0, 0)
        self.end = (9, 9)

    def test_manhattan_distance(self):
        self.assertEqual(manhattan_distance((0, 0), (9, 9)), 18)
        self.assertEqual(manhattan_distance((2, 3), (5, 7)), 7)

    def test_open_grid_shortest_path_optimality(self):
        """BFS, Dijkstra, and A* must produce optimal path length 18 on a 10x10 open grid."""
        walls = []
        bfs_res = run_algorithm("BFS", self.rows, self.cols, self.start, self.end, walls)
        dijk_res = run_algorithm("Dijkstra", self.rows, self.cols, self.start, self.end, walls)
        astar_res = run_algorithm("A*", self.rows, self.cols, self.start, self.end, walls)
        dfs_res = run_algorithm("DFS", self.rows, self.cols, self.start, self.end, walls)

        # Optimal path length from (0,0) to (9,9) on 4-directional unit grid is 18 steps
        self.assertTrue(bfs_res["success"])
        self.assertTrue(dijk_res["success"])
        self.assertTrue(astar_res["success"])
        self.assertTrue(dfs_res["success"])

        self.assertEqual(bfs_res["path_length"], 18)
        self.assertEqual(dijk_res["path_length"], 18)
        self.assertEqual(astar_res["path_length"], 18)

        # A* heuristic should typically explore fewer or equal nodes compared to BFS and Dijkstra
        self.assertLessEqual(astar_res["nodes_explored"], dijk_res["nodes_explored"])
        print(f"\n[Open Grid 10x10]")
        print(f"BFS: explored={bfs_res['nodes_explored']}, len={bfs_res['path_length']}, time={bfs_res['execution_time_ms']}ms")
        print(f"Dijkstra: explored={dijk_res['nodes_explored']}, len={dijk_res['path_length']}, time={dijk_res['execution_time_ms']}ms")
        print(f"A*: explored={astar_res['nodes_explored']}, len={astar_res['path_length']}, time={astar_res['execution_time_ms']}ms")
        print(f"DFS: explored={dfs_res['nodes_explored']}, len={dfs_res['path_length']}, time={dfs_res['execution_time_ms']}ms")

    def test_maze_with_walls(self):
        """Test on a maze with a U-shaped obstacle."""
        walls = [
            (2, 2), (3, 2), (4, 2), (5, 2),
            (5, 3), (5, 4), (5, 5),
            (4, 5), (3, 5), (2, 5),
        ]
        start = (3, 3)
        end = (0, 0)
        all_res = run_all_algorithms(self.rows, self.cols, start, end, walls)

        self.assertTrue(all_res["BFS"]["success"])
        self.assertTrue(all_res["Dijkstra"]["success"])
        self.assertTrue(all_res["A*"]["success"])
        self.assertTrue(all_res["DFS"]["success"])

        # Optimal lengths must match
        self.assertEqual(all_res["BFS"]["path_length"], all_res["Dijkstra"]["path_length"])
        self.assertEqual(all_res["BFS"]["path_length"], all_res["A*"]["path_length"])

        # Verify path connectivity
        for name, res in all_res.items():
            path = res["path"]
            self.assertEqual(tuple(path[0]), start)
            self.assertEqual(tuple(path[-1]), end)
            for i in range(len(path) - 1):
                p1, p2 = path[i], path[i + 1]
                dist = abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])
                self.assertEqual(dist, 1, f"{name} path step is not 4-directional neighbor")
                self.assertNotIn(tuple(p1), walls)

    def test_unreachable_target(self):
        """Target completely enclosed by walls should return success: False."""
        # Enclose end (9,9)
        walls = [(8, 9), (9, 8)]
        res = run_algorithm("A*", self.rows, self.cols, (0, 0), (9, 9), walls)
        self.assertFalse(res["success"])
        self.assertEqual(res["path"], [])
        self.assertEqual(res["path_length"], 0)
        self.assertGreater(res["nodes_explored"], 0)

    def test_start_equals_end(self):
        """When start == end, zero length path."""
        res = run_algorithm("A*", self.rows, self.cols, (4, 4), (4, 4), [])
        self.assertTrue(res["success"])
        self.assertEqual(res["path"], [[4, 4]])
        self.assertEqual(res["path_length"], 0)
        self.assertEqual(res["nodes_explored"], 1)

    def test_validation_errors(self):
        """Invalid bounds or placing on wall should raise ValueError."""
        # Out of bounds
        with self.assertRaises(ValueError):
            run_algorithm("BFS", 5, 5, (-1, 0), (4, 4), [])
        with self.assertRaises(ValueError):
            run_algorithm("BFS", 5, 5, (0, 0), (5, 5), [])

        # Start or end on wall
        with self.assertRaises(ValueError):
            run_algorithm("BFS", 5, 5, (0, 0), (4, 4), [(0, 0)])
        with self.assertRaises(ValueError):
            run_algorithm("BFS", 5, 5, (0, 0), (4, 4), [(4, 4)])

        # Unknown algorithm
        with self.assertRaises(ValueError):
            run_algorithm("BellmanFord", 5, 5, (0, 0), (4, 4), [])


if __name__ == "__main__":
    unittest.main()

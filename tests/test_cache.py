"""Session cache isolation, expiry, bounds, and thread safety.

The isolation tests matter because the cache key is the only thing keeping
one browser session's results out of another's.
"""

import threading
import time
import unittest

from backend.services import query_store
from backend.services.cache import TTLMemoryCache


class KeyIsolationTests(unittest.TestCase):
    def setUp(self):
        self.cache = TTLMemoryCache(max_entries=8, max_bytes=1 << 20)

    def test_identical_queries_in_different_sessions_do_not_share(self):
        self.cache.put("aaaa:query1", ["session-a"])
        self.cache.put("bbbb:query1", ["session-b"])
        self.assertEqual(self.cache.get("aaaa:query1"), ["session-a"])
        self.assertEqual(self.cache.get("bbbb:query1"), ["session-b"])

    def test_one_session_cannot_read_another(self):
        self.cache.put("aaaa:query1", ["secret"])
        self.assertIsNone(self.cache.get("bbbb:query1"))

    def test_session_ids_are_strictly_validated(self):
        self.assertTrue(query_store.valid_session_id("0" * 32))
        self.assertTrue(query_store.valid_session_id("abcdef0123456789" * 2))
        for bad in [
            None,
            "",
            "short",
            "A" * 32,             # uppercase
            "g" * 32,             # not hex
            "0" * 31,
            "0" * 33,
            "aaaa:bbbb" + "0" * 23,  # a crafted key separator
            "../../etc/passwd",
        ]:
            with self.subTest(value=bad):
                self.assertFalse(query_store.valid_session_id(bad))

    def test_a_colon_can_never_appear_in_a_valid_session_id(self):
        """This is what makes the `session:query` key scheme injection-proof."""
        self.assertFalse(query_store.valid_session_id("a" * 15 + ":" + "b" * 16))

    def test_invalid_ids_are_replaced_not_trusted(self):
        replaced = query_store.ensure_session_id("not-a-session")
        self.assertTrue(query_store.valid_session_id(replaced))
        self.assertNotEqual(replaced, "not-a-session")

    def test_valid_ids_are_preserved(self):
        original = query_store.new_session_id()
        self.assertEqual(query_store.ensure_session_id(original), original)


class ExpiryTests(unittest.TestCase):
    def test_entries_expire(self):
        cache = TTLMemoryCache(ttl_seconds=0.05)
        cache.put("s:q", ["value"])
        self.assertEqual(cache.get("s:q"), ["value"])
        time.sleep(0.08)
        self.assertIsNone(cache.get("s:q"))

    def test_per_entry_ttl_overrides_the_default(self):
        cache = TTLMemoryCache(ttl_seconds=100)
        cache.put("s:q", ["value"], ttl=0.05)
        time.sleep(0.08)
        self.assertIsNone(cache.get("s:q"))

    def test_expired_entries_release_their_bytes(self):
        cache = TTLMemoryCache(ttl_seconds=0.05)
        cache.put("s:q", ["value"], size=1000)
        self.assertEqual(cache.nbytes, 1000)
        time.sleep(0.08)
        cache.get("s:q")
        self.assertEqual(cache.nbytes, 0)


class BoundTests(unittest.TestCase):
    def test_entry_count_bound_evicts_least_recently_used(self):
        cache = TTLMemoryCache(max_entries=3, max_bytes=1 << 30)
        for index in range(3):
            cache.put(f"s:q{index}", [index])
        cache.get("s:q0")           # q0 becomes most recently used
        cache.put("s:q3", [3])      # evicts q1, the true LRU

        self.assertIsNotNone(cache.get("s:q0"))
        self.assertIsNone(cache.get("s:q1"))
        self.assertIsNotNone(cache.get("s:q2"))
        self.assertIsNotNone(cache.get("s:q3"))
        self.assertEqual(len(cache), 3)

    def test_byte_bound_is_enforced_independently(self):
        """One 3.6 MB entry and one 40 KB entry are not equivalent."""
        cache = TTLMemoryCache(max_entries=100, max_bytes=1000)
        cache.put("s:small", ["a"], size=400)
        cache.put("s:big", ["b"], size=900)
        self.assertIsNone(cache.get("s:small"))
        self.assertIsNotNone(cache.get("s:big"))
        self.assertLessEqual(cache.nbytes, 1000)

    def test_delete_releases_bytes(self):
        cache = TTLMemoryCache()
        cache.put("s:q", ["value"], size=500)
        cache.delete("s:q")
        self.assertEqual(cache.nbytes, 0)
        self.assertEqual(len(cache), 0)

    def test_overwriting_a_key_does_not_double_count(self):
        cache = TTLMemoryCache()
        cache.put("s:q", ["a"], size=500)
        cache.put("s:q", ["b"], size=700)
        self.assertEqual(cache.nbytes, 700)
        self.assertEqual(len(cache), 1)


class ConcurrencyTests(unittest.TestCase):
    """Dash runs sync callbacks in anyio's threadpool, so this is real."""

    def test_concurrent_access_leaves_the_structure_consistent(self):
        cache = TTLMemoryCache(max_entries=16, max_bytes=1 << 20)
        errors: list[BaseException] = []
        barrier = threading.Barrier(8)

        def worker(worker_id: int) -> None:
            try:
                barrier.wait()
                for index in range(200):
                    key = f"s{worker_id}:q{index % 20}"
                    cache.put(key, [worker_id, index], size=64)
                    cache.get(key)
                    if index % 7 == 0:
                        cache.delete(key)
            except BaseException as error:  # noqa: BLE001
                errors.append(error)

        threads = [
            threading.Thread(target=worker, args=(worker_id,))
            for worker_id in range(8)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        self.assertEqual(errors, [])
        self.assertLessEqual(len(cache), 16)
        self.assertGreaterEqual(cache.nbytes, 0)
        self.assertEqual(cache.nbytes, 64 * len(cache))


if __name__ == "__main__":
    unittest.main()

import unittest
from unittest.mock import patch

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.core.database import Base
from app.models.conversation import Conversation
from app.models.knowledge_base import KnowledgeBase
from app.models.message import Message
from app.models.user import User
from app.schemas.retrieval import RetrievalSearchItem
from app.services.retrieval import RetrievalPipelineResult, RetrievalTrace
from app.services.rag_service import (
    NO_ANSWER_MESSAGE,
    ask_knowledge_base,
    stream_knowledge_base_events,
)


class RagServiceTestCase(unittest.TestCase):
    def setUp(self):
        self.engine = create_engine("sqlite:///:memory:")
        TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        Base.metadata.create_all(bind=self.engine)
        self.db = TestingSessionLocal()

        user = User(
            email="tester@example.com",
            password_hash="hashed-password",
            nickname="tester",
        )
        other_user = User(
            email="other@example.com",
            password_hash="hashed-password",
            nickname="other",
        )
        self.db.add_all([user, other_user])
        self.db.commit()
        self.db.refresh(user)
        self.db.refresh(other_user)

        knowledge_base = KnowledgeBase(user_id=user.id, name="demo", description="demo kb")
        other_kb = KnowledgeBase(user_id=other_user.id, name="other", description="other kb")
        second_kb = KnowledgeBase(user_id=user.id, name="second", description="second kb")
        self.db.add_all([knowledge_base, other_kb, second_kb])
        self.db.commit()
        self.db.refresh(knowledge_base)
        self.db.refresh(other_kb)
        self.db.refresh(second_kb)

        self.user_id = user.id
        self.knowledge_base_id = knowledge_base.id
        self.other_knowledge_base_id = other_kb.id
        self.second_knowledge_base_id = second_kb.id

    def tearDown(self):
        self.db.close()
        Base.metadata.drop_all(bind=self.engine)
        self.engine.dispose()

    @staticmethod
    def _find_graph_trace_item(graph_trace, node_name):
        for item in graph_trace:
            current_node = item["node"] if isinstance(item, dict) else item.node
            if current_node == node_name:
                return item
        raise AssertionError(f"missing graph trace item: {node_name}")

    @staticmethod
    def _build_pipeline(
        final_candidates,
        *,
        dense_candidates=None,
        bm25_candidates=None,
        fused_candidates=None,
        rerank_applied=False,
    ):
        return RetrievalPipelineResult(
            dense_candidates=list(
                dense_candidates if dense_candidates is not None else final_candidates
            ),
            bm25_candidates=list(bm25_candidates or []),
            fused_candidates=list(
                fused_candidates if fused_candidates is not None else final_candidates
            ),
            final_candidates=list(final_candidates),
            trace=RetrievalTrace(
                embedding_ms=0,
                dense_candidates_count=len(
                    dense_candidates if dense_candidates is not None else final_candidates
                ),
                bm25_candidates_count=len(bm25_candidates or []),
                fusion_candidates_count=len(
                    fused_candidates if fused_candidates is not None else final_candidates
                ),
                rerank_applied=rerank_applied,
            ),
        )

    def test_returns_reject_debug_when_no_retrieval_results(self):
        with patch(
            "app.services.rag_service.run_retrieval_pipeline",
            return_value=self._build_pipeline([]),
        ):
            with patch("app.services.rag_service.generate_answer") as mocked_generate_answer:
                response = ask_knowledge_base(
                    db=self.db,
                    current_user_id=self.user_id,
                    knowledge_base_id=self.knowledge_base_id,
                    question="问题",
                    top_k=3,
                    debug=True,
                )

        mocked_generate_answer.assert_not_called()
        self.assertEqual(response.answer, NO_ANSWER_MESSAGE)
        self.assertEqual(response.citations, [])
        self.assertEqual(response.debug.decision, "reject")
        self.assertIsNone(response.debug.top1_score)
        self.assertEqual(response.debug.threshold, 0.35)
        self.assertEqual(response.debug.llm_ms, 0)
        self.assertIsNone(response.debug.final_context_preview)
        self.assertEqual(response.debug.retrieved_chunks, [])
        self.assertEqual(
            [item.node for item in response.debug.graph_trace],
            [
                "validate_request",
                "resolve_conversation",
                "rewrite_question",
                "retrieve_dense_candidates",
                "retrieve_bm25_candidates",
                "fuse_candidates",
                "rerank_candidates",
                "relevance_guard",
                "build_citations",
                "finalize_response",
            ],
        )
        self.assertEqual(response.debug.graph_trace[2].status, "skipped")
        rewrite_item = self._find_graph_trace_item(
            response.debug.graph_trace, "rewrite_question"
        )
        retrieve_item = self._find_graph_trace_item(
            response.debug.graph_trace, "retrieve_dense_candidates"
        )
        bm25_item = self._find_graph_trace_item(
            response.debug.graph_trace, "retrieve_bm25_candidates"
        )
        fuse_item = self._find_graph_trace_item(
            response.debug.graph_trace, "fuse_candidates"
        )
        rerank_item = self._find_graph_trace_item(
            response.debug.graph_trace, "rerank_candidates"
        )
        guard_item = self._find_graph_trace_item(
            response.debug.graph_trace, "relevance_guard"
        )
        citations_item = self._find_graph_trace_item(
            response.debug.graph_trace, "build_citations"
        )
        self.assertFalse(rewrite_item.used_history)
        self.assertEqual(rewrite_item.rewritten_question, "问题")
        self.assertEqual(retrieve_item.retrieval_count, 0)
        self.assertEqual(retrieve_item.dense_candidates_count, 0)
        self.assertEqual(bm25_item.bm25_candidates_count, 0)
        self.assertEqual(fuse_item.fusion_candidates_count, 0)
        self.assertFalse(rerank_item.rerank_applied)
        self.assertEqual(guard_item.decision, "reject")
        self.assertEqual(guard_item.threshold, 0.35)
        self.assertIsNone(guard_item.top1_score)
        self.assertEqual(guard_item.reject_reason, "no_candidate")
        self.assertEqual(citations_item.cited_count, 0)
        self.assertFalse(citations_item.used_fallback_citations)
        self.assertEqual(response.retrieved_chunks, [])
        self.assertIsNotNone(response.conversation_id)

        messages = (
            self.db.query(Message)
            .filter(Message.conversation_id == response.conversation_id)
            .order_by(Message.id.asc())
            .all()
        )
        self.assertEqual([message.role for message in messages], ["user", "assistant"])
        self.assertEqual(messages[1].content, NO_ANSWER_MESSAGE)

    def test_returns_answer_and_debug_info_when_model_cites_sources(self):
        retrieved = [
            RetrievalSearchItem(
                chunk_id=1,
                document_id=1,
                filename="demo.txt",
                chunk_index=0,
                start_offset=0,
                end_offset=6,
                content="第一段内容",
                score=0.9,
            ),
            RetrievalSearchItem(
                chunk_id=2,
                document_id=1,
                filename="demo.txt",
                chunk_index=1,
                start_offset=6,
                end_offset=12,
                content="第二段内容",
                score=0.8,
            ),
        ]

        with patch(
            "app.services.rag_service.run_retrieval_pipeline",
            return_value=self._build_pipeline(retrieved, rerank_applied=True),
        ):
            with patch(
                "app.services.rag_service.generate_answer",
                return_value="结论如下 [S1] 进一步说明 [S2]",
            ):
                response = ask_knowledge_base(
                    db=self.db,
                    current_user_id=self.user_id,
                    knowledge_base_id=self.knowledge_base_id,
                    question="问题",
                    top_k=3,
                    debug=True,
                )

        self.assertEqual(response.answer, "结论如下 [S1] 进一步说明 [S2]")
        self.assertEqual(len(response.citations), 2)
        self.assertEqual(response.citations[0].chunk_id, 1)
        self.assertEqual(response.citations[0].chunk_index, 0)
        self.assertEqual(response.citations[0].start_offset, 0)
        self.assertEqual(response.citations[0].end_offset, 6)
        self.assertEqual(len(response.retrieved_chunks or []), 2)
        self.assertEqual(response.debug.decision, "answer")
        self.assertEqual(response.debug.top1_score, 0.9)
        self.assertEqual(response.debug.threshold, 0.35)
        self.assertIsNotNone(response.debug.final_context_preview)
        self.assertEqual(len(response.debug.retrieved_chunks), 2)
        self.assertEqual(
            [item.node for item in response.debug.graph_trace],
            [
                "validate_request",
                "resolve_conversation",
                "rewrite_question",
                "retrieve_dense_candidates",
                "retrieve_bm25_candidates",
                "fuse_candidates",
                "rerank_candidates",
                "relevance_guard",
                "generate_answer",
                "build_citations",
                "finalize_response",
            ],
        )
        self.assertTrue(response.debug.retrieved_chunks[0].whether_cited)
        self.assertTrue(response.debug.retrieved_chunks[1].whether_cited)
        self.assertGreaterEqual(response.debug.llm_ms, 0)
        self.assertGreaterEqual(response.debug.retrieval_ms, 0)
        self.assertGreaterEqual(response.debug.total_ms, 0)
        self.assertTrue(all(item.duration_ms >= 0 for item in response.debug.graph_trace))
        rewrite_item = self._find_graph_trace_item(
            response.debug.graph_trace, "rewrite_question"
        )
        retrieve_item = self._find_graph_trace_item(
            response.debug.graph_trace, "retrieve_dense_candidates"
        )
        rerank_item = self._find_graph_trace_item(
            response.debug.graph_trace, "rerank_candidates"
        )
        guard_item = self._find_graph_trace_item(
            response.debug.graph_trace, "relevance_guard"
        )
        citations_item = self._find_graph_trace_item(
            response.debug.graph_trace, "build_citations"
        )
        self.assertFalse(rewrite_item.used_history)
        self.assertEqual(rewrite_item.rewritten_question, "问题")
        self.assertEqual(retrieve_item.retrieval_count, 2)
        self.assertEqual(retrieve_item.dense_candidates_count, 2)
        self.assertTrue(rerank_item.rerank_applied)
        self.assertEqual(rerank_item.top1_score, 0.9)
        self.assertEqual(guard_item.decision, "answer")
        self.assertEqual(guard_item.threshold, 0.35)
        self.assertEqual(guard_item.top1_score, 0.9)
        self.assertEqual(citations_item.cited_count, 2)
        self.assertFalse(citations_item.used_fallback_citations)

        conversation = self.db.query(Conversation).filter(Conversation.id == response.conversation_id).first()
        self.assertEqual(conversation.title, "问题")

        messages = (
            self.db.query(Message)
            .filter(Message.conversation_id == response.conversation_id)
            .order_by(Message.id.asc())
            .all()
        )
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0].role, "user")
        self.assertEqual(messages[1].role, "assistant")
        self.assertEqual(len(messages[1].citations_json or []), 2)
        self.assertEqual(messages[1].citations_json[0]["chunk_id"], 1)
        self.assertEqual(messages[1].citations_json[0]["start_offset"], 0)
        self.assertEqual(messages[1].citations_json[0]["end_offset"], 6)

    def test_fallback_citations_mark_only_top_three_chunks_as_cited(self):
        retrieved = [
            RetrievalSearchItem(
                chunk_id=1,
                document_id=1,
                filename="demo.txt",
                chunk_index=0,
                start_offset=0,
                end_offset=6,
                content="第一段内容",
                score=0.9,
            ),
            RetrievalSearchItem(
                chunk_id=2,
                document_id=1,
                filename="demo.txt",
                chunk_index=1,
                start_offset=6,
                end_offset=12,
                content="第二段内容",
                score=0.8,
            ),
            RetrievalSearchItem(
                chunk_id=3,
                document_id=1,
                filename="demo.txt",
                chunk_index=2,
                start_offset=12,
                end_offset=18,
                content="第三段内容",
                score=0.7,
            ),
            RetrievalSearchItem(
                chunk_id=4,
                document_id=1,
                filename="demo.txt",
                chunk_index=3,
                start_offset=18,
                end_offset=24,
                content="第四段内容",
                score=0.6,
            ),
        ]

        with patch(
            "app.services.rag_service.run_retrieval_pipeline",
            return_value=self._build_pipeline(retrieved),
        ):
            with patch(
                "app.services.rag_service.generate_answer",
                return_value="这是没有引用标记的答案",
            ):
                response = ask_knowledge_base(
                    db=self.db,
                    current_user_id=self.user_id,
                    knowledge_base_id=self.knowledge_base_id,
                    question="问题",
                    top_k=4,
                    debug=True,
                )

        cited_flags = [item.whether_cited for item in response.debug.retrieved_chunks]
        self.assertEqual(cited_flags, [True, True, True, False])
        citations_item = self._find_graph_trace_item(
            response.debug.graph_trace, "build_citations"
        )
        self.assertEqual(citations_item.cited_count, 3)
        self.assertTrue(citations_item.used_fallback_citations)

    def test_debug_is_none_when_request_does_not_enable_it(self):
        retrieved = [
            RetrievalSearchItem(
                chunk_id=1,
                document_id=1,
                filename="demo.txt",
                chunk_index=0,
                start_offset=0,
                end_offset=6,
                content="第一段内容",
                score=0.9,
            )
        ]

        with patch(
            "app.services.rag_service.run_retrieval_pipeline",
            return_value=self._build_pipeline(retrieved),
        ):
            with patch(
                "app.services.rag_service.generate_answer",
                return_value="答案 [S1]",
            ):
                response = ask_knowledge_base(
                    db=self.db,
                    current_user_id=self.user_id,
                    knowledge_base_id=self.knowledge_base_id,
                    question="问题",
                    top_k=3,
                    debug=False,
                )

        self.assertIsNone(response.debug)
        self.assertIsNone(response.retrieved_chunks)

    def test_blank_question_returns_400(self):
        with self.assertRaises(Exception) as context:
            ask_knowledge_base(
                db=self.db,
                current_user_id=self.user_id,
                knowledge_base_id=self.knowledge_base_id,
                question="   ",
                top_k=3,
                debug=False,
            )

        self.assertEqual(context.exception.status_code, 400)

    def test_reuses_existing_conversation_and_saves_messages(self):
        conversation = Conversation(
            user_id=self.user_id,
            knowledge_base_id=self.knowledge_base_id,
            title="已有会话",
        )
        self.db.add(conversation)
        self.db.commit()
        self.db.refresh(conversation)

        retrieved = [
            RetrievalSearchItem(
                chunk_id=1,
                document_id=1,
                filename="demo.txt",
                chunk_index=0,
                start_offset=0,
                end_offset=6,
                content="第一段内容",
                score=0.9,
            )
        ]

        with patch(
            "app.services.rag_service.run_retrieval_pipeline",
            return_value=self._build_pipeline(retrieved),
        ):
            with patch(
                "app.services.rag_service.generate_answer",
                return_value="答案 [S1]",
            ):
                response = ask_knowledge_base(
                    db=self.db,
                    current_user_id=self.user_id,
                    knowledge_base_id=self.knowledge_base_id,
                    question="继续问",
                    top_k=3,
                    debug=False,
                    conversation_id=conversation.id,
                )

        self.assertEqual(response.conversation_id, conversation.id)
        messages = (
            self.db.query(Message)
            .filter(Message.conversation_id == conversation.id)
            .order_by(Message.id.asc())
            .all()
        )
        self.assertEqual([message.role for message in messages], ["user", "assistant"])

    def test_rejects_conversation_from_another_knowledge_base(self):
        conversation = Conversation(
            user_id=self.user_id,
            knowledge_base_id=self.second_knowledge_base_id,
            title="另一个知识库会话",
        )
        self.db.add(conversation)
        self.db.commit()
        self.db.refresh(conversation)

        with self.assertRaises(Exception) as context:
            ask_knowledge_base(
                db=self.db,
                current_user_id=self.user_id,
                knowledge_base_id=self.knowledge_base_id,
                question="问题",
                conversation_id=conversation.id,
            )

        self.assertEqual(context.exception.status_code, 400)

    def test_rejects_unowned_conversation(self):
        conversation = Conversation(
            user_id=999,
            knowledge_base_id=self.knowledge_base_id,
            title="别人的会话",
        )
        self.db.add(conversation)
        self.db.commit()
        self.db.refresh(conversation)

        with self.assertRaises(Exception) as context:
            ask_knowledge_base(
                db=self.db,
                current_user_id=self.user_id,
                knowledge_base_id=self.knowledge_base_id,
                question="问题",
                conversation_id=conversation.id,
            )

        self.assertEqual(context.exception.status_code, 404)

    def test_existing_history_rewrites_follow_up_question_before_retrieval(self):
        conversation = Conversation(
            user_id=self.user_id,
            knowledge_base_id=self.knowledge_base_id,
            title="history",
        )
        self.db.add(conversation)
        self.db.commit()
        self.db.refresh(conversation)

        self.db.add_all(
            [
                Message(
                    conversation_id=conversation.id,
                    role="user",
                    content="请介绍一下请假制度",
                ),
                Message(
                    conversation_id=conversation.id,
                    role="assistant",
                    content="请假需要提前发起审批。",
                ),
            ]
        )
        self.db.commit()

        retrieved = [
            RetrievalSearchItem(
                chunk_id=1,
                document_id=1,
                filename="demo.txt",
                chunk_index=0,
                start_offset=0,
                end_offset=8,
                content="请假审批需要提前两天提交。",
                score=0.88,
            )
        ]

        with patch(
            "app.services.rag_service.rewrite_question",
            return_value="请假制度里请假审批需要提前多久提交？",
        ) as mocked_rewrite:
            with patch(
                "app.services.rag_service.run_retrieval_pipeline",
                return_value=self._build_pipeline(retrieved),
            ) as mocked_search:
                with patch(
                    "app.services.rag_service.generate_answer",
                    return_value="需要提前两天提交审批。 [S1]",
                ):
                    response = ask_knowledge_base(
                        db=self.db,
                        current_user_id=self.user_id,
                        knowledge_base_id=self.knowledge_base_id,
                        question="那要提前多久？",
                        top_k=3,
                        debug=True,
                        conversation_id=conversation.id,
                    )

        self.assertEqual(response.answer, "需要提前两天提交审批。 [S1]")
        mocked_rewrite.assert_called_once()
        self.assertEqual(
            mocked_search.call_args.kwargs["query"],
            "请假制度里请假审批需要提前多久提交？",
        )
        self.assertIsNotNone(response.debug.final_context_preview)
        self.assertIn("recent_turn_summary", response.debug.final_context_preview)
        rewrite_item = self._find_graph_trace_item(
            response.debug.graph_trace, "rewrite_question"
        )
        self.assertEqual(rewrite_item.node, "rewrite_question")
        self.assertEqual(rewrite_item.status, "completed")
        self.assertTrue(rewrite_item.used_history)
        self.assertEqual(
            rewrite_item.rewritten_question,
            "请假制度里请假审批需要提前多久提交？",
        )

    def test_stream_events_emit_start_delta_final_and_save_assistant_message(self):
        retrieved = [
            RetrievalSearchItem(
                chunk_id=1,
                document_id=1,
                filename="demo.txt",
                chunk_index=0,
                start_offset=0,
                end_offset=6,
                content="绗竴娈靛唴瀹?",
                score=0.9,
            )
        ]

        with patch(
            "app.services.rag_service.run_retrieval_pipeline",
            return_value=self._build_pipeline(retrieved),
        ):
            with patch(
                "app.services.rag_service.stream_answer",
                return_value=iter(["绛旀", " [S1]"]),
            ):
                events = list(
                    stream_knowledge_base_events(
                        db=self.db,
                        current_user_id=self.user_id,
                        knowledge_base_id=self.knowledge_base_id,
                        question="闂",
                        top_k=3,
                        debug=True,
                    )
        )

        self.assertEqual([name for name, _ in events], ["start", "delta", "delta", "final"])
        self.assertEqual(events[-1][1]["answer"], "绛旀 [S1]")
        self.assertEqual(
            [item["node"] for item in events[-1][1]["debug"]["graph_trace"]],
            [
                "validate_request",
                "resolve_conversation",
                "rewrite_question",
                "retrieve_dense_candidates",
                "retrieve_bm25_candidates",
                "fuse_candidates",
                "rerank_candidates",
                "relevance_guard",
                "stream_answer",
                "build_citations",
                "finalize_response",
            ],
        )
        retrieve_item = self._find_graph_trace_item(
            events[-1][1]["debug"]["graph_trace"], "retrieve_dense_candidates"
        )
        rerank_item = self._find_graph_trace_item(
            events[-1][1]["debug"]["graph_trace"], "rerank_candidates"
        )
        guard_item = self._find_graph_trace_item(
            events[-1][1]["debug"]["graph_trace"], "relevance_guard"
        )
        citations_item = self._find_graph_trace_item(
            events[-1][1]["debug"]["graph_trace"], "build_citations"
        )
        self.assertEqual(retrieve_item["retrieval_count"], 1)
        self.assertEqual(retrieve_item["dense_candidates_count"], 1)
        self.assertFalse(rerank_item["rerank_applied"])
        self.assertEqual(rerank_item["top1_score"], 0.9)
        self.assertEqual(guard_item["decision"], "answer")
        self.assertEqual(guard_item["threshold"], 0.35)
        self.assertEqual(guard_item["top1_score"], 0.9)
        self.assertEqual(citations_item["cited_count"], 1)
        self.assertFalse(citations_item["used_fallback_citations"])
        messages = (
            self.db.query(Message)
            .filter(Message.conversation_id == events[-1][1]["conversation_id"])
            .order_by(Message.id.asc())
            .all()
        )
        self.assertEqual([message.role for message in messages], ["user", "assistant"])
        self.assertEqual(messages[-1].content, "绛旀 [S1]")

    def test_stream_error_does_not_save_assistant_message(self):
        retrieved = [
            RetrievalSearchItem(
                chunk_id=1,
                document_id=1,
                filename="demo.txt",
                chunk_index=0,
                start_offset=0,
                end_offset=6,
                content="绗竴娈靛唴瀹?",
                score=0.9,
            )
        ]

        with patch(
            "app.services.rag_service.run_retrieval_pipeline",
            return_value=self._build_pipeline(retrieved),
        ):
            with patch(
                "app.services.rag_service.stream_answer",
                side_effect=RuntimeError("llm unavailable"),
            ):
                with self.assertRaises(RuntimeError):
                    list(
                        stream_knowledge_base_events(
                            db=self.db,
                            current_user_id=self.user_id,
                            knowledge_base_id=self.knowledge_base_id,
                            question="闂",
                            top_k=3,
                            debug=False,
                        )
                    )

        messages = self.db.query(Message).order_by(Message.id.asc()).all()
        self.assertEqual([message.role for message in messages], ["user"])

    def test_debug_items_include_extended_retrieval_fields(self):
        retrieved = [
            RetrievalSearchItem(
                chunk_id=1,
                document_id=1,
                filename="demo.txt",
                chunk_index=0,
                start_offset=0,
                end_offset=6,
                content="扩展字段片段",
                score=0.91,
                guard_score=0.88,
                source_channels=["dense", "bm25", "rerank"],
                dense_score=0.77,
                bm25_score=6.3,
                fusion_score=0.52,
                rerank_score=0.97,
                dense_rank=2,
                bm25_rank=1,
                fusion_rank=1,
                rerank_rank=1,
            )
        ]

        with patch(
            "app.services.rag_service.run_retrieval_pipeline",
            return_value=self._build_pipeline(retrieved, rerank_applied=True),
        ):
            with patch(
                "app.services.rag_service.generate_answer",
                return_value="扩展字段答案 [S1]",
            ):
                response = ask_knowledge_base(
                    db=self.db,
                    current_user_id=self.user_id,
                    knowledge_base_id=self.knowledge_base_id,
                    question="测试扩展字段",
                    top_k=3,
                    debug=True,
                )

        top_level_item = response.retrieved_chunks[0]
        debug_item = response.debug.retrieved_chunks[0]
        self.assertEqual(top_level_item.guard_score, 0.88)
        self.assertEqual(top_level_item.source_channels, ["dense", "bm25", "rerank"])
        self.assertEqual(top_level_item.dense_score, 0.77)
        self.assertEqual(top_level_item.bm25_score, 6.3)
        self.assertEqual(top_level_item.fusion_score, 0.52)
        self.assertEqual(top_level_item.rerank_score, 0.97)
        self.assertEqual(top_level_item.dense_rank, 2)
        self.assertEqual(top_level_item.bm25_rank, 1)
        self.assertEqual(top_level_item.fusion_rank, 1)
        self.assertEqual(top_level_item.rerank_rank, 1)
        self.assertEqual(debug_item.guard_score, 0.88)
        self.assertEqual(debug_item.source_channels, ["dense", "bm25", "rerank"])
        self.assertEqual(debug_item.dense_score, 0.77)
        self.assertEqual(debug_item.bm25_score, 6.3)
        self.assertEqual(debug_item.fusion_score, 0.52)
        self.assertEqual(debug_item.rerank_score, 0.97)
        self.assertEqual(debug_item.dense_rank, 2)
        self.assertEqual(debug_item.bm25_rank, 1)
        self.assertEqual(debug_item.fusion_rank, 1)
        self.assertEqual(debug_item.rerank_rank, 1)

    def test_reject_response_uses_fixed_no_answer_message(self):
        with patch(
            "app.services.rag_service.run_retrieval_pipeline",
            return_value=self._build_pipeline([]),
        ):
            response = ask_knowledge_base(
                db=self.db,
                current_user_id=self.user_id,
                knowledge_base_id=self.knowledge_base_id,
                question="没有命中",
                top_k=3,
                debug=True,
            )

        self.assertEqual(NO_ANSWER_MESSAGE, "当前知识库中未找到足够相关内容。")
        self.assertEqual(response.answer, "当前知识库中未找到足够相关内容。")
        messages = (
            self.db.query(Message)
            .filter(Message.conversation_id == response.conversation_id)
            .order_by(Message.id.asc())
            .all()
        )
        self.assertEqual(messages[-1].content, "当前知识库中未找到足够相关内容。")

    def test_stream_and_non_stream_share_graph_sequence(self):
        retrieved = [
            RetrievalSearchItem(
                chunk_id=1,
                document_id=1,
                filename="demo.txt",
                chunk_index=0,
                start_offset=0,
                end_offset=6,
                content="同一条链路",
                score=0.9,
                guard_score=0.9,
            )
        ]

        with patch(
            "app.services.rag_service.run_retrieval_pipeline",
            return_value=self._build_pipeline(retrieved, rerank_applied=True),
        ):
            with patch(
                "app.services.rag_service.generate_answer",
                return_value="同步答案 [S1]",
            ):
                response = ask_knowledge_base(
                    db=self.db,
                    current_user_id=self.user_id,
                    knowledge_base_id=self.knowledge_base_id,
                    question="同步测试",
                    top_k=3,
                    debug=True,
                )

        with patch(
            "app.services.rag_service.run_retrieval_pipeline",
            return_value=self._build_pipeline(retrieved, rerank_applied=True),
        ):
            with patch(
                "app.services.rag_service.stream_answer",
                return_value=iter(["同步答案", " [S1]"]),
            ):
                events = list(
                    stream_knowledge_base_events(
                        db=self.db,
                        current_user_id=self.user_id,
                        knowledge_base_id=self.knowledge_base_id,
                        question="同步测试",
                        top_k=3,
                        debug=True,
                    )
                )

        non_stream_nodes = [item.node for item in response.debug.graph_trace]
        stream_nodes = [item["node"] for item in events[-1][1]["debug"]["graph_trace"]]
        normalized_non_stream = [
            "answer" if node == "generate_answer" else node for node in non_stream_nodes
        ]
        normalized_stream = [
            "answer" if node == "stream_answer" else node for node in stream_nodes
        ]
        self.assertEqual(normalized_non_stream, normalized_stream)

"""
Unified Rainbow Strategy
v5: Exploration space is 7 × 4 (category × role) only.
    - archive.select_seed/select_target_cell 모두 severity 무관 uniform sampling
    - select_seed      → (category, role, instruction)
    - select_target_cell → (category, role)
    - mutator/refiner/trigger 호출 시 severity 인자 전달 X
    - strategy key: {category}-{role}  (severity sub-key 없음)
    - learn_from_result: judge의 자연어 reasoning만 전달
"""

import numpy as np
from typing import Tuple
from harm_archive import HarmArchive
from harm_mutator import HarmMutator
from harm_instruction_refiner import HarmInstructionRefiner
from harm_trigger_agent import HarmTriggerAgent
from dotenv import load_dotenv
load_dotenv()


class UnifiedRainbowStrategy:
    def __init__(
        self,
        archive: HarmArchive,
        mutator: HarmMutator,
        refiner: HarmInstructionRefiner,
        trigger_agent: HarmTriggerAgent,
        mutation_strategy: str = "adaptive",
        learning_rate: float = 1.0,
    ):
        self.archive           = archive
        self.mutator           = mutator
        self.refiner           = refiner
        self.trigger_agent     = trigger_agent
        self.mutation_strategy = mutation_strategy
        self.learning_rate     = learning_rate

        self.stats = {
            "total_instructions": 0,
            "mutation_types": {"category": 0, "role": 0, "crossover": 0, "random": 0},
            "learning_applied": 0,
            "new_generations":  0,
        }

        self.learning_buffer = []
        self.buffer_size     = 20

    # ── main entry ────────────────────────────────────────────────────────────

    async def get_next_instruction(
        self,
        patient_profile: dict,
        conversation_context: list,
        turn_number: int = 1,
    ) -> Tuple[str, str, str, dict]:
        """
        Returns:
            (category, role, instruction, metadata)

        severity는 agent 체인 어디에도 노출되지 않음.
        archive sampling 역시 severity 무관 (uniform).
        """
        self.stats["total_instructions"] += 1

        if self.archive.is_empty():
            print("📝 Archive empty — generating new...")
            return await self._generate_new_instruction(patient_profile, conversation_context)

        # 1. Seed / Target — severity 무관 sampling
        seed_cat, seed_role, seed_inst = self.archive.select_seed()
        target_cat, target_role        = self.archive.select_target_cell()

        print(
            f"🌈 Rainbow: Seed ({seed_cat}, {seed_role}) → Target ({target_cat}, {target_role})"
        )

        # 2. Mutation type
        coverage      = self.archive.get_archive_summary()["coverage"]
        mutation_type = self._select_mutation_type(coverage)
        self.stats["mutation_types"][mutation_type] += 1

        # 3. 관련 전략 (category × role 매칭만)
        strategies = self._get_relevant_strategies(target_cat, target_role)
        if strategies:
            self.stats["learning_applied"] += 1
            print(f"🧠 Applying {len(strategies)} learned strategies")

        # 4. Mutation
        instruction = await self._mutate_with_strategies(
            mutation_type        = mutation_type,
            seed_instruction     = seed_inst,
            seed_category        = seed_cat,
            seed_role            = seed_role,
            target_category      = target_cat,
            target_role          = target_role,
            strategies           = strategies,
            patient_profile      = patient_profile,
            conversation_context = conversation_context,
        )

        metadata = {
            "method":             "unified_rainbow",
            "mutation_type":      mutation_type,
            "seed_category":      seed_cat,
            "seed_role":          seed_role,
            "target_category":    target_cat,
            "target_role":        target_role,
            "strategies_applied": len(strategies),
            "coverage":           coverage,
        }

        return target_cat, target_role, instruction, metadata

    # ── strategy retrieval ────────────────────────────────────────────────────

    def _get_relevant_strategies(
        self,
        target_cat: str,
        target_role: str,
    ) -> list:
        strategies = []

        key = f"{target_cat}-{target_role}"
        if key in self.refiner.accumulated_strategies:
            match = self.refiner.accumulated_strategies[key]
            n     = max(1, int(len(match) * self.learning_rate))
            strategies.extend(match[:n])

        # 최근 buffer (severity 필터 없음)
        strategies.extend(self.learning_buffer[-3:])

        return strategies

    # ── mutation dispatch ─────────────────────────────────────────────────────

    async def _mutate_with_strategies(
        self,
        mutation_type:        str,
        seed_instruction:     str,
        seed_category:        str,
        seed_role:            str,
        target_category:      str,
        target_role:          str,
        strategies:           list,
        patient_profile:      dict,
        conversation_context: list,
    ) -> str:
        strategy_context = self._build_strategy_context(strategies)

        if mutation_type == "category":
            return await self.mutator.mutate_category(
                seed_instruction       = seed_instruction,
                seed_category          = seed_category,
                seed_role              = seed_role,
                target_category        = target_category,
                patient_profile        = patient_profile,
                conversation_context   = conversation_context,
                accumulated_strategies = strategy_context,
            )
        elif mutation_type == "role":
            return await self.mutator.mutate_role(
                seed_instruction       = seed_instruction,
                seed_category          = seed_category,
                seed_role              = seed_role,
                target_role            = target_role,
                patient_profile        = patient_profile,
                conversation_context   = conversation_context,
                accumulated_strategies = strategy_context,
            )
        elif mutation_type == "crossover":
            try:
                p2_cat, p2_role, p2_inst = self.archive.select_seed()
            except ValueError:
                p2_cat, p2_role, p2_inst = seed_category, seed_role, seed_instruction
            return await self.mutator.crossover_mutation(
                instruction1           = seed_instruction,
                category1              = seed_category,
                role1                  = seed_role,
                instruction2           = p2_inst,
                category2              = p2_cat,
                role2                  = p2_role,
                target_category        = target_category,
                target_role            = target_role,
                patient_profile        = patient_profile,
                conversation_context   = conversation_context,
                accumulated_strategies = strategy_context,
            )
        else:  # random
            return await self.mutator.random_mutation(
                seed_instruction       = seed_instruction,
                seed_category          = seed_category,
                seed_role              = seed_role,
                target_category        = target_category,
                target_role            = target_role,
                patient_profile        = patient_profile,
                conversation_context   = conversation_context,
                accumulated_strategies = strategy_context,
            )

    def _build_strategy_context(self, strategies: list) -> str:
        if not strategies:
            return ""
        context = "\n\nLEARNED STRATEGIES (apply these):\n"
        for i, strat in enumerate(strategies[:5], 1):
            if isinstance(strat, dict) and "bullets" in strat:
                context += f"\nStrategy {i}:\n"
                for bullet in strat["bullets"]:
                    context += f"- {bullet}\n"
        return context

    # ── learning ──────────────────────────────────────────────────────────────

    async def learn_from_result(
        self,
        category:           str,
        role:               str,
        instruction:        str,
        counselor_response: str,
        patient_response:   str,
        reasoning:          str,
    ):
        """
        judge의 자연어 reasoning만으로 학습.
        severity 점수는 여기 전달 X — refiner는 오직 reasoning text만 봄.
        """
        print(f"🧠 Learning from judge reasoning ({category} × {role})...")

        bullets = await self.refiner.analyze_failure(
            category             = category,
            role                 = role,
            original_instruction = instruction,
            counselor_response   = counselor_response,
            failure_reason       = reasoning,
        )

        key = f"{category}-{role}"

        entry = {
            "bullets":              bullets,
            "original_instruction": instruction,
            "reasoning":            reasoning,
        }

        if key not in self.refiner.accumulated_strategies:
            self.refiner.accumulated_strategies[key] = []
        self.refiner.accumulated_strategies[key].insert(0, entry)
        self.refiner.accumulated_strategies[key] = \
            self.refiner.accumulated_strategies[key][:10]

        self.learning_buffer.append({
            "category":  category,
            "role":      role,
            "reasoning": reasoning,
            "bullets":   bullets,
        })
        if len(self.learning_buffer) > self.buffer_size:
            self.learning_buffer.pop(0)

    # ── helpers ───────────────────────────────────────────────────────────────

    def _select_mutation_type(self, coverage: float) -> str:
        if self.mutation_strategy == "exploration_first":
            weights = [0.45, 0.45, 0.05, 0.05] if coverage < 0.5 else [0.30, 0.30, 0.25, 0.15]
        elif self.mutation_strategy == "balanced":
            weights = [0.35, 0.35, 0.20, 0.10]
        else:  # adaptive
            if coverage < 0.3:
                weights = [0.45, 0.45, 0.05, 0.05]
            elif coverage < 0.7:
                weights = [0.35, 0.35, 0.20, 0.10]
            else:
                weights = [0.25, 0.25, 0.30, 0.20]
        return np.random.choice(["category", "role", "crossover", "random"], p=weights)

    async def _generate_new_instruction(
        self,
        patient_profile: dict,
        conversation_context: list,
    ) -> Tuple[str, str, str, dict]:
        self.stats["new_generations"] += 1
        cat, role, inst = await self.trigger_agent.generate_harm_trigger(
            patient_profile, conversation_context
        )
        return cat, role, inst, {"method": "new_generation"}

    def get_statistics(self) -> dict:
        total = self.stats["total_instructions"]
        return {
            "total_instructions":    total,
            "mutation_distribution": {
                k: v / total if total > 0 else 0
                for k, v in self.stats["mutation_types"].items()
            },
            "learning_applied": self.stats["learning_applied"],
            "learning_rate": (
                self.stats["learning_applied"] / total if total > 0 else 0
            ),
        }

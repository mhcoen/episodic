#!/usr/bin/env python3
"""
Modular test harness for conversational drift detection using real conversation data.

This script provides a configurable framework for testing drift detection
with pluggable data loading and analysis strategies.
"""

import sys
import os
from .ml import ConversationalDrift
from .ml.testing import TestingStrategy, AnalysisStrategy
from typing import List, Dict, Any, Optional
import json


class DriftTestHarness:
    """Modular test harness for analyzing drift in real conversations."""
    
    def __init__(
        self,
        drift_config: Optional[Dict[str, Any]] = None,
        testing_strategy: str = "recent_conversations",
        analysis_strategy: str = "basic_stats",
        verbose: bool = True,
        **strategy_kwargs
    ):
        """
        Initialize the modular test harness.
        
        Args:
            drift_config: Configuration for ConversationalDrift (embedding provider, etc.)
            testing_strategy: Data loading strategy ("recent_conversations", "conversation_chains", etc.)
            analysis_strategy: Analysis approach ("basic_stats", "detailed_transitions", etc.)
            verbose: Whether to print detailed output during testing
            **strategy_kwargs: Additional parameters for testing and analysis strategies
        """
        self.verbose = verbose
        
        # Initialize drift calculator with configuration
        if drift_config:
            self.drift_calculator = ConversationalDrift(**drift_config)
        else:
            self.drift_calculator = ConversationalDrift()
        
        # Initialize modular strategies
        testing_kwargs = {k: v for k, v in strategy_kwargs.items() if k.startswith('testing_')}
        analysis_kwargs = {k: v for k, v in strategy_kwargs.items() if k.startswith('analysis_')}
        
        # Remove prefixes from kwargs
        testing_kwargs = {k[8:]: v for k, v in testing_kwargs.items()}  # Remove 'testing_' prefix
        analysis_kwargs = {k[9:]: v for k, v in analysis_kwargs.items()}  # Remove 'analysis_' prefix
        
        self.testing_strategy = TestingStrategy(testing_strategy, **testing_kwargs)
        self.analysis_strategy = AnalysisStrategy(analysis_strategy, **analysis_kwargs)
        
    def log(self, message: str):
        """Log a message if verbose mode is enabled."""
        if self.verbose:
            print(message)
    
    def run_test(self, text_field: str = "content") -> Dict[str, Any]:
        """
        Run the complete test using configured strategies.
        
        Args:
            text_field: Field containing the text content to analyze
            
        Returns:
            Complete test results dictionary
        """
        self.log(f"Running drift test with strategy: {self.testing_strategy.strategy}")
        self.log(f"Test description: {self.testing_strategy.get_description()}")
        
        # Load test data using the configured strategy
        conversation_sequences = self.testing_strategy.load_test_data()
        
        if not conversation_sequences:
            return {"error": "No conversation data loaded by testing strategy"}
        
        self.log(f"✓ Loaded {len(conversation_sequences)} conversation sequences")
        
        # Analyze using the configured analysis strategy
        self.log(f"Analyzing with strategy: {self.analysis_strategy.strategy}")
        analysis_results = self.analysis_strategy.analyze_drift_results(
            self.drift_calculator,
            conversation_sequences,
            text_field
        )
        
        # Add test metadata
        test_results = {
            "test_metadata": {
                "testing_strategy": self.testing_strategy.strategy,
                "analysis_strategy": self.analysis_strategy.strategy,
                "drift_calculator_config": {
                    "embedding_provider": self.drift_calculator.embedding_provider.provider,
                    "embedding_model": self.drift_calculator.embedding_provider.model,
                    "distance_algorithm": self.drift_calculator.distance_function.algorithm,
                    "peak_strategy": self.drift_calculator.peak_detector.strategy
                },
                "sequences_loaded": len(conversation_sequences),
                "text_field": text_field
            },
            "analysis_results": analysis_results
        }
        
        return test_results
    
    def print_results(self, results: Dict[str, Any]):
        """
        Print test results in a formatted way.
        
        Args:
            results: Results dictionary from run_test()
        """
        if "error" in results:
            self.log(f"❌ Test failed: {results['error']}")
            return
        
        metadata = results["test_metadata"]
        analysis = results["analysis_results"]
        
        self.log("\n" + "=" * 60)
        self.log("📊 DRIFT TEST RESULTS")
        self.log("=" * 60)
        
        # Test configuration
        self.log(f"Testing Strategy: {metadata['testing_strategy']}")
        self.log(f"Analysis Strategy: {metadata['analysis_strategy']}")
        self.log(f"Sequences Analyzed: {metadata['sequences_loaded']}")
        
        config = metadata["drift_calculator_config"]
        self.log(f"Embeddings: {config['embedding_provider']} ({config['embedding_model']})")
        self.log(f"Distance: {config['distance_algorithm']}")
        self.log(f"Peak Detection: {config['peak_strategy']}")
        
        # Analysis-specific results
        if "error" in analysis:
            self.log(f"\n❌ Analysis failed: {analysis['error']}")
        else:
            analysis_type = analysis.get("analysis_type", "unknown")
            
            if analysis_type == "basic_stats":
                self._print_basic_stats(analysis)
            elif analysis_type == "detailed_transitions":
                self._print_detailed_transitions(analysis)
            elif analysis_type == "peak_analysis":
                self._print_peak_analysis(analysis)
            elif analysis_type == "comparative":
                self._print_comparative_analysis(analysis)
        
        self.log("=" * 60)
    
    def _print_basic_stats(self, analysis: Dict[str, Any]):
        """Print basic statistics analysis results."""
        stats = analysis["overall_stats"]
        
        self.log(f"\n📈 Overall Statistics:")
        self.log(f"  Total Nodes: {stats['total_nodes']}")
        self.log(f"  Total Transitions: {stats['total_transitions']}")
        self.log(f"  Average Drift: {stats['overall_avg_drift']:.3f}")
        self.log(f"  Max Drift: {stats['overall_max_drift']:.3f}")
        self.log(f"  Min Drift: {stats['overall_min_drift']:.3f}")
        self.log(f"  Standard Deviation: {stats['drift_std_dev']:.3f}")
        
        # Show sequence summaries
        self.log(f"\n📋 Sequence Summaries:")
        for seq in analysis["sequence_summaries"][:5]:  # Show first 5
            self.log(f"  Sequence {seq['sequence_id']}: {seq['node_count']} nodes, "
                    f"avg={seq['avg_drift']:.3f}, peaks={seq['peak_count']}")
    
    def _print_detailed_transitions(self, analysis: Dict[str, Any]):
        """Print detailed transitions analysis results."""
        self.log(f"\n🔍 Top Drift Transitions (showing first 5):")
        
        for i, transition in enumerate(analysis["transitions"][:5]):
            self.log(f"\n  #{i+1} - Drift: {transition['drift_score']:.3f}")
            self.log(f"    {transition['from_node_id']} ({transition['from_role']}) → "
                    f"{transition['to_node_id']} ({transition['to_role']})")
            self.log(f"    From: {transition['from_content']}")
            self.log(f"    To:   {transition['to_content']}")
    
    def _print_peak_analysis(self, analysis: Dict[str, Any]):
        """Print peak analysis results."""
        stats = analysis["peak_statistics"]
        
        self.log(f"\n⛰️  Peak Analysis:")
        self.log(f"  Total Peaks: {stats['total_peaks']}")
        self.log(f"  Significant Peaks (≥{analysis['highlight_threshold']}): {stats['significant_peaks']}")
        self.log(f"  Peak Density: {stats['peak_density']:.3f}")
        
        self.log(f"\n🏔️  Top Peaks:")
        for i, peak in enumerate(analysis["peaks"][:5]):
            significance = "⚡ SIGNIFICANT" if peak["is_significant"] else "  normal"
            self.log(f"  #{i+1} [{significance}] Node {peak['peak_node_id']}: {peak['drift_score']:.3f}")
            self.log(f"      {peak['peak_content']}")
    
    def _print_comparative_analysis(self, analysis: Dict[str, Any]):
        """Print comparative analysis results."""
        stats = analysis["comparison_stats"]
        
        self.log(f"\n📊 Comparative Analysis (baseline: {stats['baseline_drift']}):")
        self.log(f"  Above Baseline: {stats['above_baseline_count']} ({stats['above_baseline_pct']:.1f}%)")
        self.log(f"  Below Baseline: {stats['below_baseline_count']}")
        self.log(f"  Average Above: {stats['avg_above_baseline']:.3f}")
        self.log(f"  Average Below: {stats['avg_below_baseline']:.3f}")
        self.log(f"  Overall Average: {stats['overall_avg']:.3f}")
        self.log(f"  Deviation from Baseline: {stats['deviation_from_baseline']:.3f}")
        self.log(f"  Classification: {analysis['classification']}")


def main():
    """Main test harness execution with multiple test configurations."""
    print("🧪 Modular Episodic Conversational Drift Test Harness")
    print("=" * 60)
    
    # Test 1: Basic functionality with recent conversations
    print("\n1️⃣  BASIC FUNCTIONALITY TEST")
    harness1 = DriftTestHarness(
        testing_strategy="recent_conversations",
        analysis_strategy="basic_stats",
        testing_limit=8
    )
    results1 = harness1.run_test()
    harness1.print_results(results1)
    
    # Test 2: Detailed transition analysis on conversation chains
    print("\n\n2️⃣  DETAILED TRANSITION ANALYSIS")
    harness2 = DriftTestHarness(
        testing_strategy="conversation_chains",
        analysis_strategy="detailed_transitions",
        testing_max_chains=2,
        testing_min_length=3,
        analysis_content_limit=80
    )
    results2 = harness2.run_test()
    harness2.print_results(results2)
    
    # Test 3: Peak analysis with different drift configuration
    print("\n\n3️⃣  PEAK ANALYSIS TEST")
    harness3 = DriftTestHarness(
        drift_config={
            "distance_algorithm": "euclidean",
            "peak_strategy": "relative",
            "min_prominence": 0.15
        },
        testing_strategy="recent_conversations",
        analysis_strategy="peak_analysis",
        testing_limit=10,
        analysis_highlight_threshold=0.6
    )
    results3 = harness3.run_test()
    harness3.print_results(results3)
    
    # Test 4: Comparative analysis
    print("\n\n4️⃣  COMPARATIVE ANALYSIS")
    harness4 = DriftTestHarness(
        testing_strategy="conversation_chains",
        analysis_strategy="comparative",
        testing_max_chains=3,
        analysis_baseline_drift=0.5
    )
    results4 = harness4.run_test()
    harness4.print_results(results4)
    
    # Summary
    print("\n\n" + "=" * 60)
    print("🎯 MODULAR TEST HARNESS SUMMARY")
    print("=" * 60)
    
    successful_tests = 0
    total_tests = 4
    
    for i, results in enumerate([results1, results2, results3, results4], 1):
        if "error" not in results and "error" not in results.get("analysis_results", {}):
            print(f"✅ Test {i}: PASSED")
            successful_tests += 1
        else:
            print(f"❌ Test {i}: FAILED")
    
    print(f"\nResults: {successful_tests}/{total_tests} tests passed")
    
    if successful_tests == total_tests:
        print("🎉 All modular tests passed! The drift detection system is fully functional.")
    elif successful_tests > 0:
        print("⚠️  Some tests passed. System is partially functional.")
    else:
        print("🔧 All tests failed. Check your configuration and database.")
    
    print("=" * 60)


if __name__ == "__main__":
    main()
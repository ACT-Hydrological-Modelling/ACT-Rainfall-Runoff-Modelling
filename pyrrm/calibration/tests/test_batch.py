"""
Tests for the batch experiment runner.

Tests:
- ExperimentList construction and combinations()
- ExperimentList.from_dicts() factory
- BatchExperimentRunner accepting ExperimentList
- Logging file creation (batch.log, per-experiment logs)
- Output directory structure (results/, logs/, summary files)
- Resume from new directory structure
- Backward-compatible resume from flat directory
- YAML config with 'experiments' key parsing
"""

import json
import os
import shutil
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from pyrrm.models import GR4J
from pyrrm.calibration.objective_functions import NSE, KGE
from pyrrm.calibration.batch import (
    DEFAULT_CATCHMENT,
    make_experiment_key,
    make_apex_tags,
    parse_experiment_key,
    ExperimentSpec,
    ExperimentGrid,
    ExperimentList,
    BatchExperimentRunner,
    BatchResult,
    get_model_class,
    _make_run_dir_name,
    _find_latest_run_dir,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def tmp_output_dir(tmp_path):
    """Provide a temporary output directory."""
    d = tmp_path / 'batch_output'
    d.mkdir()
    return d


@pytest.fixture
def synthetic_data():
    """Create minimal synthetic data for batch tests."""
    np.random.seed(42)
    n = 800
    dates = pd.date_range('2000-01-01', periods=n, freq='D')
    precip = np.maximum(0, np.random.exponential(3, n))
    pet = 3.0 + 2.0 * np.sin(2 * np.pi * np.arange(n) / 365.25)
    pet = np.maximum(0.1, pet)

    inputs_df = pd.DataFrame(
        {'precipitation': precip, 'pet': pet}, index=dates,
    )
    model = GR4J({'X1': 350, 'X2': 0.5, 'X3': 90, 'X4': 1.7})
    result_df = model.run(inputs_df)
    observed = np.maximum(0, result_df['flow'].values)
    return inputs_df, observed


@pytest.fixture
def sample_grid():
    """A small 1x1x1 experiment grid for quick tests."""
    return ExperimentGrid(
        models={'GR4J': GR4J()},
        objectives={'nse': NSE()},
        algorithms={'sceua': {'method': 'sceua_direct', 'max_evals': 200, 'seed': 42}},
    )


@pytest.fixture
def sample_spec():
    """A single ExperimentSpec."""
    return ExperimentSpec(
        key='gr4j_nse_sceua',
        model_name='GR4J',
        model=GR4J(),
        objective_name='nse',
        objective=NSE(),
        algorithm_name='sceua',
        algorithm_kwargs={'method': 'sceua_direct', 'max_evals': 200, 'seed': 42},
    )


# ============================================================================
# ExperimentList tests
# ============================================================================

class TestExperimentList:
    """Tests for ExperimentList class."""

    def test_construction(self, sample_spec):
        exp_list = ExperimentList([sample_spec])
        assert len(exp_list) == 1

    def test_combinations_returns_copy(self, sample_spec):
        exp_list = ExperimentList([sample_spec])
        result = exp_list.combinations()
        assert len(result) == 1
        assert result[0].key == 'gr4j_nse_sceua'
        assert result is not exp_list.specs

    def test_multiple_specs(self, sample_spec):
        spec2 = ExperimentSpec(
            key='gr4j_kge_sceua',
            model_name='GR4J',
            model=GR4J(),
            objective_name='kge',
            objective=KGE(),
            algorithm_name='sceua',
            algorithm_kwargs={'method': 'sceua_direct', 'max_evals': 200},
        )
        exp_list = ExperimentList([sample_spec, spec2])
        assert len(exp_list) == 2
        keys = [s.key for s in exp_list.combinations()]
        assert keys == ['gr4j_nse_sceua', 'gr4j_kge_sceua']

    def test_from_dicts_basic(self):
        dicts = [
            {
                'key': 'gr4j_nse',
                'model': 'GR4J',
                'objective': {'type': 'NSE'},
                'algorithm': {'method': 'sceua_direct', 'max_evals': 100},
            },
        ]
        exp_list = ExperimentList.from_dicts(dicts)
        assert len(exp_list) == 1
        spec = exp_list.specs[0]
        assert spec.key == 'gr4j_nse'
        assert spec.model_name == 'GR4J'
        assert spec.objective_name == 'nse'
        assert spec.algorithm_kwargs['max_evals'] == 100

    def test_from_dicts_multiple(self):
        dicts = [
            {
                'key': 'gr4j_nse',
                'model': 'GR4J',
                'objective': {'type': 'NSE'},
                'algorithm': {'method': 'sceua_direct', 'max_evals': 100},
            },
            {
                'key': 'gr4j_kge',
                'model': 'GR4J',
                'objective': {'type': 'KGE'},
                'algorithm': {'method': 'sceua_direct', 'max_evals': 200},
            },
        ]
        exp_list = ExperimentList.from_dicts(dicts)
        assert len(exp_list) == 2
        assert exp_list.specs[0].key == 'gr4j_nse'
        assert exp_list.specs[1].key == 'gr4j_kge'

    def test_from_dicts_with_model_params(self):
        dicts = [
            {
                'key': 'gr4j_custom',
                'model': 'GR4J',
                'model_params': {},
                'objective': 'NSE',
                'algorithm': {'method': 'sceua_direct', 'max_evals': 50},
            },
        ]
        exp_list = ExperimentList.from_dicts(dicts)
        assert len(exp_list) == 1

    def test_from_dicts_unknown_model_raises(self):
        dicts = [
            {
                'key': 'bad',
                'model': 'NonExistentModel',
                'objective': {'type': 'NSE'},
                'algorithm': {'method': 'sceua_direct'},
            },
        ]
        with pytest.raises(ValueError, match="Unknown model"):
            ExperimentList.from_dicts(dicts)


# ============================================================================
# Run-dir naming tests
# ============================================================================

class TestRunDirNaming:
    def test_make_run_dir_name_no_label(self):
        name = _make_run_dir_name()
        parts = name.split('_')
        assert len(parts) >= 3
        assert len(parts[0]) == 8  # YYYYMMDD
        assert len(parts[1]) == 6  # HHMMSS
        assert len(parts[2]) == 4  # hex id

    def test_make_run_dir_name_with_label(self):
        name = _make_run_dir_name('my experiment!')
        assert 'my_experiment_' in name

    def test_find_latest_run_dir_empty(self, tmp_output_dir):
        assert _find_latest_run_dir(tmp_output_dir) is None

    def test_find_latest_run_dir(self, tmp_output_dir):
        d1 = tmp_output_dir / '20260101_100000_aaaa'
        d1.mkdir()
        (d1 / 'results').mkdir()

        d2 = tmp_output_dir / '20260102_100000_bbbb'
        d2.mkdir()
        (d2 / 'results').mkdir()

        latest = _find_latest_run_dir(tmp_output_dir)
        assert latest is not None
        assert latest.name == '20260102_100000_bbbb'


# ============================================================================
# BatchExperimentRunner with ExperimentList
# ============================================================================

class TestBatchRunnerWithExperimentList:
    def test_accepts_experiment_list(self, synthetic_data, sample_spec, tmp_output_dir):
        inputs_df, observed = synthetic_data
        exp_list = ExperimentList([sample_spec])
        runner = BatchExperimentRunner(
            inputs=inputs_df,
            observed=observed,
            grid=exp_list,
            output_dir=str(tmp_output_dir),
            warmup_period=365,
            progress_bar=False,
        )
        result = runner.run(resume=False, run_name='test_list')
        assert len(result.results) == 1
        assert sample_spec.key in result.results
        assert result.run_dir is not None


# ============================================================================
# Output directory structure tests
# ============================================================================

class TestOutputStructure:
    def test_run_creates_timestamped_folder(self, synthetic_data, sample_grid, tmp_output_dir):
        inputs_df, observed = synthetic_data
        runner = BatchExperimentRunner(
            inputs=inputs_df,
            observed=observed,
            grid=sample_grid,
            output_dir=str(tmp_output_dir),
            warmup_period=365,
            progress_bar=False,
        )
        result = runner.run(resume=False, run_name='structure_test')
        run_dir = Path(result.run_dir)
        assert run_dir.exists()
        assert run_dir.parent == tmp_output_dir

    def test_results_subfolder(self, synthetic_data, sample_grid, tmp_output_dir):
        inputs_df, observed = synthetic_data
        runner = BatchExperimentRunner(
            inputs=inputs_df,
            observed=observed,
            grid=sample_grid,
            output_dir=str(tmp_output_dir),
            warmup_period=365,
            progress_bar=False,
        )
        result = runner.run(resume=False)
        run_dir = Path(result.run_dir)
        results_dir = run_dir / 'results'
        assert results_dir.exists()
        pkl_files = list(results_dir.glob('*.pkl'))
        assert len(pkl_files) == 1
        assert pkl_files[0].stem == 'catchment_gr4j_nse_sceua'

    def test_logs_subfolder(self, synthetic_data, sample_grid, tmp_output_dir):
        inputs_df, observed = synthetic_data
        runner = BatchExperimentRunner(
            inputs=inputs_df,
            observed=observed,
            grid=sample_grid,
            output_dir=str(tmp_output_dir),
            warmup_period=365,
            progress_bar=False,
        )
        result = runner.run(resume=False)
        run_dir = Path(result.run_dir)
        logs_dir = run_dir / 'logs'
        assert logs_dir.exists()
        log_files = list(logs_dir.glob('*.log'))
        assert len(log_files) == 1
        assert log_files[0].stem == 'catchment_gr4j_nse_sceua'

    def test_batch_log_file(self, synthetic_data, sample_grid, tmp_output_dir):
        inputs_df, observed = synthetic_data
        runner = BatchExperimentRunner(
            inputs=inputs_df,
            observed=observed,
            grid=sample_grid,
            output_dir=str(tmp_output_dir),
            warmup_period=365,
            progress_bar=False,
        )
        result = runner.run(resume=False)
        run_dir = Path(result.run_dir)
        batch_log = run_dir / 'batch.log'
        assert batch_log.exists()
        content = batch_log.read_text()
        assert 'BATCH RUN COMPLETE' in content

    def test_summary_files(self, synthetic_data, sample_grid, tmp_output_dir):
        inputs_df, observed = synthetic_data
        runner = BatchExperimentRunner(
            inputs=inputs_df,
            observed=observed,
            grid=sample_grid,
            output_dir=str(tmp_output_dir),
            warmup_period=365,
            progress_bar=False,
        )
        result = runner.run(resume=False)
        run_dir = Path(result.run_dir)

        json_path = run_dir / 'batch_summary.json'
        assert json_path.exists()
        with open(json_path) as f:
            summary = json.load(f)
        assert summary['completed'] == 1
        assert summary['failed'] == 0

        csv_path = run_dir / 'batch_summary.csv'
        assert csv_path.exists()
        df = pd.read_csv(csv_path)
        assert len(df) == 1

    def test_config_snapshot(self, synthetic_data, sample_grid, tmp_output_dir):
        inputs_df, observed = synthetic_data
        runner = BatchExperimentRunner(
            inputs=inputs_df,
            observed=observed,
            grid=sample_grid,
            output_dir=str(tmp_output_dir),
            warmup_period=365,
            progress_bar=False,
        )
        result = runner.run(resume=False)
        run_dir = Path(result.run_dir)
        config_files = list(run_dir.glob('config.*'))
        assert len(config_files) == 1

    def test_batch_result_pkl(self, synthetic_data, sample_grid, tmp_output_dir):
        inputs_df, observed = synthetic_data
        runner = BatchExperimentRunner(
            inputs=inputs_df,
            observed=observed,
            grid=sample_grid,
            output_dir=str(tmp_output_dir),
            warmup_period=365,
            progress_bar=False,
        )
        result = runner.run(resume=False)
        run_dir = Path(result.run_dir)
        pkl_path = run_dir / 'batch_result.pkl'
        assert pkl_path.exists()
        reloaded = BatchResult.load(str(pkl_path))
        assert len(reloaded.results) == 1


# ============================================================================
# Resume tests
# ============================================================================

class TestResume:
    def test_resume_skips_completed(self, synthetic_data, sample_grid, tmp_output_dir):
        inputs_df, observed = synthetic_data
        runner = BatchExperimentRunner(
            inputs=inputs_df,
            observed=observed,
            grid=sample_grid,
            output_dir=str(tmp_output_dir),
            warmup_period=365,
            progress_bar=False,
        )
        result1 = runner.run(resume=False, run_name='run1')
        assert len(result1.results) == 1

        result2 = runner.run(resume=True)
        assert len(result2.results) == 1
        assert result2.runtime_seconds == 0.0

    def test_resume_from_explicit_path(self, synthetic_data, sample_grid, tmp_output_dir):
        inputs_df, observed = synthetic_data
        runner = BatchExperimentRunner(
            inputs=inputs_df,
            observed=observed,
            grid=sample_grid,
            output_dir=str(tmp_output_dir),
            warmup_period=365,
            progress_bar=False,
        )
        result1 = runner.run(resume=False, run_name='explicit')
        run_dir_1 = result1.run_dir

        result2 = runner.run(resume_from=run_dir_1)
        assert len(result2.results) == 1
        assert result2.runtime_seconds == 0.0

    def test_resume_from_nonexistent_raises(self, synthetic_data, sample_grid, tmp_output_dir):
        inputs_df, observed = synthetic_data
        runner = BatchExperimentRunner(
            inputs=inputs_df,
            observed=observed,
            grid=sample_grid,
            output_dir=str(tmp_output_dir),
            warmup_period=365,
            progress_bar=False,
        )
        with pytest.raises(FileNotFoundError):
            runner.run(resume_from='/nonexistent/path')


# ============================================================================
# Backward compatibility resume (flat .pkl layout)
# ============================================================================

class TestBackwardCompatResume:
    def test_legacy_flat_pkl_resume(self, synthetic_data, sample_grid, tmp_output_dir):
        inputs_df, observed = synthetic_data
        runner = BatchExperimentRunner(
            inputs=inputs_df,
            observed=observed,
            grid=sample_grid,
            output_dir=str(tmp_output_dir),
            warmup_period=365,
            progress_bar=False,
        )
        result = runner.run(resume=False)
        run_dir = Path(result.run_dir)

        new_key = 'catchment_gr4j_nse_sceua'
        src_pkl = run_dir / 'results' / f'{new_key}.pkl'
        dst_pkl = tmp_output_dir / f'{new_key}.pkl'
        shutil.copy2(str(src_pkl), str(dst_pkl))

        shutil.rmtree(str(run_dir))

        runner2 = BatchExperimentRunner(
            inputs=inputs_df,
            observed=observed,
            grid=sample_grid,
            output_dir=str(tmp_output_dir),
            warmup_period=365,
            progress_bar=False,
        )
        result2 = runner2.run(resume=True)
        assert len(result2.results) == 1
        assert result2.runtime_seconds == 0.0


# ============================================================================
# YAML / JSON config parsing
# ============================================================================

class TestConfigParsing:
    def test_json_grid_config(self, synthetic_data, tmp_output_dir):
        inputs_df, observed = synthetic_data
        config = {
            'models': {'GR4J': {}},
            'objectives': {'nse': {'type': 'NSE'}},
            'algorithms': {'sceua': {'method': 'sceua_direct', 'max_evals': 200, 'seed': 42}},
            'warmup_days': 365,
            'output_dir': str(tmp_output_dir),
            'backend': 'sequential',
            'progress_bar': False,
        }
        config_path = tmp_output_dir / 'config.json'
        with open(config_path, 'w') as f:
            json.dump(config, f)

        runner = BatchExperimentRunner.from_config(
            str(config_path), inputs_df, observed,
        )
        assert isinstance(runner.grid, ExperimentGrid)
        result = runner.run(resume=False)
        assert len(result.results) == 1

    def test_json_experiment_list_config(self, synthetic_data, tmp_output_dir):
        inputs_df, observed = synthetic_data
        config = {
            'experiments': [
                {
                    'key': 'gr4j_nse',
                    'model': 'GR4J',
                    'objective': {'type': 'NSE'},
                    'algorithm': {'method': 'sceua_direct', 'max_evals': 200, 'seed': 42},
                },
            ],
            'warmup_days': 365,
            'output_dir': str(tmp_output_dir),
            'backend': 'sequential',
            'progress_bar': False,
        }
        config_path = tmp_output_dir / 'config.json'
        with open(config_path, 'w') as f:
            json.dump(config, f)

        runner = BatchExperimentRunner.from_config(
            str(config_path), inputs_df, observed,
        )
        assert isinstance(runner.grid, ExperimentList)
        result = runner.run(resume=False)
        assert len(result.results) == 1

    def test_json_both_modes_raises(self, synthetic_data, tmp_output_dir):
        inputs_df, observed = synthetic_data
        config = {
            'models': {'GR4J': {}},
            'objectives': {'nse': {'type': 'NSE'}},
            'algorithms': {'sceua': {'method': 'sceua_direct'}},
            'experiments': [
                {'key': 'x', 'model': 'GR4J', 'objective': {'type': 'NSE'},
                 'algorithm': {'method': 'sceua_direct'}},
            ],
        }
        config_path = tmp_output_dir / 'config.json'
        with open(config_path, 'w') as f:
            json.dump(config, f)

        with pytest.raises(ValueError, match="not both"):
            BatchExperimentRunner.from_config(
                str(config_path), inputs_df, observed,
            )


# ============================================================================
# BatchResult tests
# ============================================================================

class TestBatchResult:
    def test_save_default_path(self, synthetic_data, sample_grid, tmp_output_dir):
        inputs_df, observed = synthetic_data
        runner = BatchExperimentRunner(
            inputs=inputs_df,
            observed=observed,
            grid=sample_grid,
            output_dir=str(tmp_output_dir),
            warmup_period=365,
            progress_bar=False,
        )
        result = runner.run(resume=False)
        assert result.run_dir is not None

        pkl_path = Path(result.run_dir) / 'batch_result.pkl'
        assert pkl_path.exists()
        reloaded = BatchResult.load(str(pkl_path))
        assert reloaded.run_dir == result.run_dir

    def test_save_explicit_path(self, tmp_path):
        br = BatchResult(run_dir=None)
        path = str(tmp_path / 'test_result.pkl')
        br.save(path)
        assert Path(path).exists()
        reloaded = BatchResult.load(path)
        assert isinstance(reloaded, BatchResult)

    def test_save_no_path_no_run_dir_raises(self):
        br = BatchResult(run_dir=None)
        with pytest.raises(ValueError, match="No path given"):
            br.save()

    def test_repr(self):
        br = BatchResult(runtime_seconds=10.5, run_dir='/tmp/test')
        text = repr(br)
        assert '0 completed' in text
        assert '10.5s' in text
        assert '/tmp/test' in text


# ============================================================================
# Experiment key helpers
# ============================================================================

class TestMakeExperimentKey:
    def test_basic_four_fields(self):
        key = make_experiment_key('GR4J', 'nse', 'sceua', catchment='410734')
        assert key == '410734_gr4j_nse_sceua'

    def test_default_catchment(self):
        key = make_experiment_key('Sacramento', 'kge', 'dream')
        assert key == f'{DEFAULT_CATCHMENT}_sacramento_kge_dream'

    def test_with_transformation(self):
        key = make_experiment_key('GR4J', 'nse', 'sceua',
                                  catchment='410734', transformation='sqrt')
        assert key == '410734_gr4j_nse_sqrt_sceua'

    def test_with_extra_tags(self):
        tags = make_apex_tags(dynamics_strength=0.5, regime_emphasis='uniform')
        key = make_experiment_key('Sacramento', 'apex', 'sceua',
                                  catchment='410734', transformation='sqrt',
                                  extra_tags=tags)
        assert key == '410734_sacramento_apex_sqrt-k05-uniform_sceua'

    def test_sanitises_case_and_special_chars(self):
        key = make_experiment_key('GR4J', 'NSE', 'SCE-UA', catchment='Test 1')
        assert key == 'test1_gr4j_nse_sceua'

    def test_extra_tags_without_transformation(self):
        tags = ['k03', 'uniform']
        key = make_experiment_key('Sacramento', 'apex', 'sceua',
                                  catchment='410734', extra_tags=tags)
        assert key == '410734_sacramento_apex_none-k03-uniform_sceua'


class TestMakeApexTags:
    def test_defaults(self):
        tags = make_apex_tags()
        assert tags == ['k05', 'uniform']

    def test_non_default_kappa(self):
        tags = make_apex_tags(dynamics_strength=0.3)
        assert tags == ['k03', 'uniform']

    def test_non_default_regime(self):
        tags = make_apex_tags(regime_emphasis='low_flow')
        assert tags == ['k05', 'lowflow']

    def test_all_non_default(self):
        tags = make_apex_tags(dynamics_strength=0.7, regime_emphasis='balanced')
        assert tags == ['k07', 'balanced']


class TestParseExperimentKey:
    def test_four_fields(self):
        parsed = parse_experiment_key('410734_gr4j_nse_sceua')
        assert parsed == {
            'catchment': '410734',
            'model': 'gr4j',
            'objective': 'nse',
            'algorithm': 'sceua',
        }

    def test_five_fields_with_transformation(self):
        parsed = parse_experiment_key('410734_sacramento_kge_sqrt_dream')
        assert parsed['transformation'] == 'sqrt'
        assert parsed['algorithm'] == 'dream'
        assert 'apex_tags' not in parsed

    def test_apex_tags(self):
        parsed = parse_experiment_key(
            '410734_sacramento_apex_sqrt-k05-uniform_sceua'
        )
        assert parsed['transformation'] == 'sqrt'
        assert parsed['algorithm'] == 'sceua'
        assert parsed['apex_tags'] == ['k05', 'uniform']

    def test_roundtrip(self):
        tags = make_apex_tags(dynamics_strength=0.3, regime_emphasis='low_flow')
        key = make_experiment_key(
            'Sacramento', 'apex', 'sceua',
            catchment='410734', transformation='log', extra_tags=tags,
        )
        parsed = parse_experiment_key(key)
        assert parsed['catchment'] == '410734'
        assert parsed['model'] == 'sacramento'
        assert parsed['objective'] == 'apex'
        assert parsed['algorithm'] == 'sceua'
        assert parsed['transformation'] == 'log'
        assert parsed['apex_tags'] == ['k03', 'lowflow']

    def test_short_key(self):
        parsed = parse_experiment_key('a_b_c')
        assert parsed == {}


class TestExperimentGridCatchment:
    def test_grid_uses_default_catchment(self):
        grid = ExperimentGrid(
            models={'GR4J': GR4J()},
            objectives={'nse': NSE()},
            algorithms={'sceua': {'method': 'sceua_direct'}},
        )
        specs = grid.combinations()
        assert specs[0].key == f'{DEFAULT_CATCHMENT}_gr4j_nse_sceua'

    def test_grid_uses_custom_catchment(self):
        grid = ExperimentGrid(
            models={'GR4J': GR4J()},
            objectives={'nse': NSE()},
            algorithms={'sceua': {'method': 'sceua_direct'}},
            catchment='410734',
        )
        specs = grid.combinations()
        assert specs[0].key == '410734_gr4j_nse_sceua'

    def test_runner_forwards_catchment_to_grid(self, synthetic_data, tmp_output_dir):
        inputs_df, observed = synthetic_data
        grid = ExperimentGrid(
            models={'GR4J': GR4J()},
            objectives={'nse': NSE()},
            algorithms={'sceua': {'method': 'sceua_direct', 'max_evals': 200, 'seed': 42}},
        )
        runner = BatchExperimentRunner(
            inputs=inputs_df,
            observed=observed,
            grid=grid,
            output_dir=str(tmp_output_dir),
            warmup_period=365,
            progress_bar=False,
            catchment='synthetic',
        )
        assert grid.catchment == 'synthetic'


class TestExperimentListAutoKey:
    def test_auto_generated_key(self):
        dicts = [
            {
                'model': 'GR4J',
                'objective': {'type': 'NSE'},
                'algorithm': {'method': 'sceua_direct'},
            },
        ]
        exp_list = ExperimentList.from_dicts(dicts, catchment='demo')
        assert exp_list.specs[0].key == 'demo_gr4j_nse_sceuadirect'

    def test_explicit_key_preserved(self):
        dicts = [
            {
                'key': 'my_custom_key',
                'model': 'GR4J',
                'objective': {'type': 'NSE'},
                'algorithm': {'method': 'sceua_direct'},
            },
        ]
        exp_list = ExperimentList.from_dicts(dicts)
        assert exp_list.specs[0].key == 'my_custom_key'

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the BSD-style license in LICENSE.
"""Parse the restart canary's legacy CLI without constructing a trainer."""
import argparse
import os


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-step', type=int, required=True)
    parser.add_argument('--resume-n', type=int, required=True)
    parser.add_argument('--data-cache', required=True)
    parser.add_argument('--hf-assets', required=True)
    args = parser.parse_args()
    os.environ['BLENDCORPUS_RESUME_CONSUMED_SAMPLES'] = str(args.resume_n)
    from torchtitan.experiments.ezpz import train
    raw = [
        '--module=ezpz.agpt', '--config=agpt_20b_real',
        f'--hf-assets-path={args.hf_assets}', '--dump-folder=/unused/preflight',
        '--checkpoint.enable', '--checkpoint.folder=/unused/preflight/checkpoints',
        f'--checkpoint.interval={args.source_step + 3}',
        '--checkpoint.keep-latest-k=0', '--checkpoint.no-last-save-model-only',
        '--checkpoint.async-mode=disabled', '--dataloader.dataset=blendcorpus',
        f'--dataloader.train-iters={args.resume_n // 48 + 4}',
        '--dataloader.dataset-path=torchtitan/experiments/ezpz/data-lists/aurora/olmo-mix-1124.txt',
        f'--dataloader.data-cache-path={args.data_cache}', '--dataloader.num-workers=0',
        '--validator.no-enable', '--metrics.no-enable-wandb',
        '--optimizer', 'sophiag', '--optimizer.lr=2.28e-5',
        '--lr-scheduler.decay-ratio=0.0', '--lr-scheduler.min-lr-factor=1.0',
        '--lr-scheduler.warmup-steps=20', '--training.seq-len=512',
        '--training.local-batch-size=1', '--training.global-batch-size=48',
        f'--training.steps={args.source_step + 3}',
        '--parallelism.tensor-parallel-degree=1',
        '--parallelism.data-parallel-replicate-degree=1',
        '--parallelism.data-parallel-shard-degree=48',
        '--parallelism.pipeline-parallel-degree=1',
        '--parallelism.context-parallel-degree=1',
    ]
    translated = train._inject_default_module_and_config(train._translate_legacy_args(raw))
    name, overrides, translated = train._extract_optimizer_args(translated)
    config = train.ConfigManager().parse_args(translated)
    assert name == 'sophiag'
    config.optimizer = train._build_optimizer_config(name, config.optimizer, overrides)
    assert config.dataloader.resume_consumed_samples == args.resume_n
    assert config.training.steps == args.source_step + 3
    assert config.training.global_batch_size == 48
    assert config.training.local_batch_size == 1
    assert name == 'sophiag'
    assert config.checkpoint.last_save_model_only is False
    print('LEGACY_RESTART_CLI_PASS', name, args.source_step, args.resume_n)


if __name__ == '__main__':
    main()

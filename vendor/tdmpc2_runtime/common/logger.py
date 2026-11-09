import dataclasses
import os
import datetime
import re
import json

import numpy as np
import pandas as pd
from termcolor import colored

from common import TASK_SET


def _local_metric(value):
	if isinstance(value, dict):
		return {key: _local_metric(item) for key, item in value.items()}
	if isinstance(value, (list, tuple)):
		return [_local_metric(item) for item in value]
	if hasattr(value, "detach"):
		return _local_metric(value.detach().cpu().tolist())
	if isinstance(value, np.ndarray):
		return _local_metric(value.tolist())
	if isinstance(value, np.generic):
		return _local_metric(value.item())
	if isinstance(value, float) and not np.isfinite(value):
		return None
	if value is None or isinstance(value, (str, int, float, bool)):
		return value
	return str(value)


CONSOLE_FORMAT = [
	("iteration", "I", "int"),
	("episode", "E", "int"),
	("step", "I", "int"),
	("episode_reward", "R", "float"),
	("episode_success", "S", "float"),
	("Reason", "Reason", "str"),
	("elapsed_time", "T", "time"),
]

CAT_TO_COLOR = {
	"pretrain": "yellow",
	"train": "blue",
	"eval": "green",
}


def make_dir(dir_path):
	"""Create directory if it does not already exist."""
	try:
		os.makedirs(dir_path)
	except OSError:
		pass
	return dir_path


def print_run(cfg):
	"""
	Pretty-printing of current run information.
	Logger calls this method at initialization.
	"""
	prefix, color, attrs = "  ", "green", ["bold"]

	def _limstr(s, maxlen=36):
		return str(s[:maxlen]) + "..." if len(str(s)) > maxlen else s

	def _pprint(k, v):
		print(
			prefix + colored(f'{k.capitalize()+":":<15}', color, attrs=attrs), _limstr(v)
		)

	observations  = ", ".join([str(v) for v in cfg.obs_shape.values()])
	kvs = [
		("task", cfg.task_title),
		("steps", f"{int(cfg.steps):,}"),
		("observations", observations),
		("actions", cfg.action_dim),
		("experiment", cfg.exp_name),
	]
	w = np.max([len(_limstr(str(kv[1]))) for kv in kvs]) + 25
	div = "-" * w
	print(div)
	for k, v in kvs:
		_pprint(k, v)
	print(div)


def cfg_to_group(cfg, return_list=False):
	"""
	Return a wandb-safe group name for logging.
	Optionally returns group name as list.
	"""
	lst = [cfg.task, re.sub("[^0-9a-zA-Z]+", "-", cfg.exp_name)]
	return lst if return_list else "-".join(lst)


class VideoRecorder:
	"""Utility class for logging evaluation videos."""

	def __init__(self, cfg, wandb, fps=15):
		self.cfg = cfg
		self._save_dir = make_dir(cfg.work_dir / 'eval_video')
		self._wandb = wandb
		self.fps = fps
		self.frames = []
		self.enabled = False

	def init(self, env, enabled=True):
		self.frames = []
		self.enabled = self._save_dir and self._wandb and enabled
		self.record(env)

	def record(self, env):
		if self.enabled:
			self.frames.append(env.render())

	def save(self, step, key='videos/eval_video'):
		if self.enabled and len(self.frames) > 0:
			frames = np.stack(self.frames)
			return self._wandb.log(
				{key: self._wandb.Video(frames.transpose(0, 3, 1, 2), fps=self.fps, format='mp4')}, step=step
			)


class Logger:
	"""Primary logging object. Logs either locally or using wandb."""

	def __init__(self, cfg):
		self._log_dir = make_dir(cfg.work_dir)
		self._model_dir = make_dir(self._log_dir / "models")
		self._save_csv = cfg.save_csv
		self._save_agent = cfg.save_agent
		self._artifact_logging_enabled = bool(cfg.get("wandb_log_artifacts", True))
		self._group = cfg_to_group(cfg)
		self._seed = cfg.seed
		self._eval = []
		self._metrics_path = self._log_dir / "metrics.jsonl"
		self._wandb_metrics_enabled = True
		print_run(cfg)
		self.project = cfg.get("wandb_project", "none")
		self.entity = cfg.get("wandb_entity", "none")
		if not cfg.enable_wandb or self.project == "none" or self.entity == "none":
			print(colored("Wandb disabled.", "blue", attrs=["bold"]))
			cfg.save_agent = False
			cfg.save_video = False
			self._wandb = None
			self._video = None
			return
		os.environ["WANDB_SILENT"] = "true" if cfg.wandb_silent else "false"
		import wandb
		run_name = cfg.get("wandb_run_name") or f"{cfg.exp_name}-seed{cfg.seed}"

		options = dict(
			project=self.project, entity=self.entity, name=str(run_name), group=self._group,
			tags=cfg_to_group(cfg, return_list=True) + [f"seed:{cfg.seed}"],
			dir=self._log_dir, config=dataclasses.asdict(cfg),
			settings=wandb.Settings(console="off", init_timeout=60),
		)
		try:
			wandb.init(**options)
		except Exception as error:
			print(f"[wandb] Online initialization failed: {error}; trying offline logging.", flush=True)
			try:
				wandb.finish(quiet=True)
			except Exception:
				pass
			try:
				wandb.init(**options, mode="offline")
			except Exception as offline_error:
				print(f"[wandb] Offline initialization failed: {offline_error}; using local metrics.jsonl.", flush=True)
				self._wandb = self._video = None
				return
		print(colored("Logs will be synced with wandb.", "blue", attrs=["bold"]))
		self._wandb = wandb
		self._video = (
			VideoRecorder(cfg, self._wandb)
			if self._wandb and cfg.save_video
			else None
		)

	@property
	def video(self):
		return self._video

	@property
	def model_dir(self):
		return self._model_dir

	def save_agent(self, agent=None, identifier='final'):
		if self._save_agent and agent:
			fp = self._model_dir / f'{str(identifier)}.pt'
			temporary = fp.with_suffix(".pt.tmp")
			agent.save(temporary)
			os.replace(temporary, fp)
			if self._wandb and self._artifact_logging_enabled:
				try:
					artifact = self._wandb.Artifact(
						self._group + '-' + str(self._seed) + '-' + str(identifier),
						type='model',
					)
					artifact.add_file(fp)
					self._wandb.log_artifact(artifact)
				except Exception as error:
					self._artifact_logging_enabled = False
					print(colored(
						f"[wandb] Model artifact upload failed: {error}. "
						f"Local checkpoint is saved at {fp}. "
						"Further artifact uploads disabled; training and metric logging continue.",
						"yellow",
					), flush=True)

	def finish(self, agent=None):
		try:
			self.save_agent(agent)
		except Exception as e:
			print(colored(f"Failed to save model: {e}", "red"))
		if self._wandb:
			try:
				self._wandb.finish()
			except Exception as error:
				print(f"[wandb] Final sync failed: {error}; local results are preserved.", flush=True)

	def _format(self, key, value, ty):
		if ty == "int":
			return f'{colored(key+":", "blue")} {int(value):,}'
		elif ty == "float":
			return f'{colored(key+":", "blue")} {value:.01f}'
		elif ty == "time":
			value = str(datetime.timedelta(seconds=int(value)))
			return f'{colored(key+":", "blue")} {value}'
		elif ty == "str":
			return f'{colored(key+":", "blue")} {value}'
		else:
			raise f"invalid log format type: {ty}"

	def _print(self, d, category):
		category = colored(category, CAT_TO_COLOR[category])
		pieces = [f" {category:<14}"]
		for k, disp_k, ty in CONSOLE_FORMAT:
			if k in d:
				pieces.append(f"{self._format(disp_k, d[k], ty):<22}")
		print("   ".join(pieces), flush=True)

	def pprint_multitask(self, d, cfg):
		"""Pretty-print evaluation metrics for multi-task training."""
		print(colored(f'Evaluated agent on {len(cfg.tasks)} tasks:', 'yellow', attrs=['bold']))
		dmcontrol_reward = []
		metaworld_reward = []
		metaworld_success = []
		for k, v in d.items():
			if '+' not in k:
				continue
			task = k.split('+')[1]
			if task in TASK_SET['mt30'] and k.startswith('episode_reward'): # DMControl
				dmcontrol_reward.append(v)
				print(colored(f'  {task:<22}\tR: {v:.01f}', 'yellow'))
			elif task in TASK_SET['mt80'] and task not in TASK_SET['mt30']: # Meta-World
				if k.startswith('episode_reward'):
					metaworld_reward.append(v)
				elif k.startswith('episode_success'):
					metaworld_success.append(v)
					print(colored(f'  {task:<22}\tS: {v:.02f}', 'yellow'))
		dmcontrol_reward = np.nanmean(dmcontrol_reward)
		d['episode_reward+avg_dmcontrol'] = dmcontrol_reward
		print(colored(f'  {"dmcontrol":<22}\tR: {dmcontrol_reward:.01f}', 'yellow', attrs=['bold']))
		if cfg.task == 'mt80':
			metaworld_reward = np.nanmean(metaworld_reward)
			metaworld_success = np.nanmean(metaworld_success)
			d['episode_reward+avg_metaworld'] = metaworld_reward
			d['episode_success+avg_metaworld'] = metaworld_success
			print(colored(f'  {"metaworld":<22}\tR: {metaworld_reward:.01f}', 'yellow', attrs=['bold']))
			print(colored(f'  {"metaworld":<22}\tS: {metaworld_success:.02f}', 'yellow', attrs=['bold']))

	def log(self, d, category="train"):
		assert category in CAT_TO_COLOR.keys(), f"invalid category: {category}"
		with self._metrics_path.open("a", encoding="utf-8") as stream:
			stream.write(json.dumps(_local_metric({"category": category, **d}), allow_nan=False) + "\n")
		if self._wandb and self._wandb_metrics_enabled:
			if category in {"train", "eval"}:
				xkey = "step"
			elif category == "pretrain":
				xkey = "iteration"
			_d = dict()
			for k, v in d.items():
				_d[category + "/" + k] = v
			try:
				self._wandb.log(_d, step=d[xkey])
			except Exception as error:
				self._wandb_metrics_enabled = False
				if self._video is not None:
					self._video._wandb = None
				print(f"[wandb] Metric upload failed: {error}; continuing with local metrics.jsonl.", flush=True)
		if category == "eval" and self._save_csv:
			keys = ["step", "episode_reward"]
			self._eval.append(np.array([d[keys[0]], d[keys[1]]]))
			pd.DataFrame(np.array(self._eval)).to_csv(
				self._log_dir / "eval.csv", header=keys, index=None
			)
		self._print(d, category)

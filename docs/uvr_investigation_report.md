# UVR.py全機能調査レポート

## 1. MainWindowクラスの概要

UVRのメインGUIは`MainWindow`クラス（1272行目〜）で実装され、Tkinterベースで221のメソッドを持つ大規模なクラスです。

### 主要コンポーネント
- **ウィンドウサイズ**: 1080p対応の固定サイズレイアウト
- **フレーム構成**: 
  - メインタイトル（バナー画像）
  - ファイルパス選択エリア
  - オプション設定エリア
  - 処理ボタンエリア
  - プログレスバー
  - コンソール出力エリア

## 2. MainWindowクラスの全メソッド（主要機能）

### GUI構築関連
- `__init__()`: クラス初期化、GUI構築
- `fill_main_frame()`: メインフレーム構築
- `fill_filePaths_Frame()`: ファイルパス選択UI構築
- `fill_options_Frame()`: オプション設定UI構築
- `bind_widgets()`: ウィジェットイベントバインド

### 設定保存・読み込み
- `load_saved_vars(data)`: 保存された設定を読み込み
- `save_values()`: 設定をJSONファイルに保存
- `auto_save()`: 自動保存機能

### 音声処理制御
- `process_initialize()`: 処理開始前チェック
- `process_start()`: 音声処理メイン
- `process_end()`: 処理終了処理
- `process_update_progress()`: 進捗バー更新
- `confirm_stop_process()`: 処理停止確認

### モデル管理
- `update_available_models()`: 利用可能モデル更新
- `selection_action()`: モデル選択アクション
- `assemble_model_data()`: モデルデータ組み立て
- `cached_source_*()`: キャッシュ管理

## 3. モデル設定詳細パラメータ

### VR（Vocal Remover）アーキテクチャ
```python
# 主要パラメータ
'aggression_setting': 0-20 (デフォルト: 5)
'window_size': 512, 1024 (デフォルト: 512)
'is_tta': TTA (Test Time Augmentation) 有効/無効
'is_post_process': ポストプロセス有効/無効
'is_high_end_process': 高域処理有効/無効
'post_process_threshold': 0.0-1.0 (デフォルト: 0.2)
```

### MDX-Net アーキテクチャ
```python
# 主要パラメータ
'mdx_segment_size': 256, 512, 1024 (デフォルト: 256)
'overlap_mdx': 0.25, 0.5, 0.75, 0.99 (デフォルト: 0.25)
'overlap_mdx23': 8, 16, 32 (MDX23専用)
'compensate': Auto選択または手動値
'mdx_batch_size': バッチサイズ
'chunks': チャンクサイズ
'margin': マージン設定
```

### Demucs アーキテクチャ
```python
# 主要パラメータ
'segment': Default, 1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50
'overlap': 0.25, 0.5, 0.75, 0.99
'shifts': シフト回数 (デフォルト: 2)
'chunks_demucs': チャンクサイズ
'margin_demucs': マージン (デフォルト: 44100)
'is_split_mode': 分割モード有効/無効
'is_demucs_combine_stems': ステム結合有効/無効
```

## 4. 音声処理オプション

### ファイル形式設定
- **対応形式**: WAV, FLAC, MP3
- **MP3品質**: 320k (デフォルト)
- **サンプリングレート**: 自動検出

### バッチ処理
- **複数ファイル対応**: ディレクトリ一括処理
- **ファイル種別**: .wav, .flac, .mp3, .m4a等
- **出力パス設定**: 個別指定可能

### GPU/CPU設定
- **GPU利用**: CUDA, DirectML対応
- **デバイス選択**: 複数GPU環境対応
- **OpenCL**: macOS/Linux対応

## 5. エンサンブル機能

### エンサンブルモード
```python
# エンサンブル設定
'is_save_all_outputs_ensemble': True  # 全出力保存
'is_append_ensemble_name': False      # 名前追加
'chosen_ensemble_var': エンサンブル方式選択
'ensemble_main_stem_var': メインステム選択
'ensemble_type_var': MAX_MIN方式
```

### 対応アルゴリズム
- **Max Spec**: 最大スペクトラム
- **Min Spec**: 最小スペクトラム  
- **Audio Average**: 音声平均
- **Manual Ensemble**: 手動エンサンブル

## 6. セカンダリモデル機能

### VRセカンダリモデル
```python
'vr_voc_inst_secondary_model': ボーカル/楽器用
'vr_other_secondary_model': その他用
'vr_bass_secondary_model': ベース用
'vr_drums_secondary_model': ドラム用
'vr_is_secondary_model_activate': 有効/無効
# スケール設定 (0.1-1.0)
'vr_voc_inst_secondary_model_scale': 0.9
'vr_other_secondary_model_scale': 0.7
'vr_bass_secondary_model_scale': 0.5
'vr_drums_secondary_model_scale': 0.5
```

### MDX/Demucsセカンダリモデル
- 同様の構造でMDX、Demucs専用設定が存在
- 各ステム別にモデルとスケール設定可能

## 7. 音声分析・テスト機能

### オーディオツール
```python
'chosen_audio_tool': AUDIO_TOOL_OPTIONS
- Manual Ensemble: 手動エンサンブル
- Time Stretch: 時間伸縮
- Change Pitch: ピッチ変更
- Align Inputs: 入力アライメント
- Match Inputs: 入力マッチング
```

### 分析パラメータ
```python
'time_stretch_rate': 2.0      # 時間伸縮率
'pitch_rate': 2.0             # ピッチ変更率
'semitone_shift': '0'         # セミトーン シフト
'is_time_correction': True    # 時間補正
'is_testing_audio': False     # テストモード
```

## 8. 設定保存・読み込み機能

### 設定ファイル
- **保存場所**: `data.json`
- **自動保存**: 設定変更時に自動保存
- **設定セット**: プリセット保存・読み込み機能

### 設定項目（部分）
```python
DEFAULT_DATA = {
    # プロセス方式
    'chosen_process_method': 'MDX-Net',
    
    # ファイル設定
    'save_format': 'WAV',
    'mp3_bit_set': '320k',
    
    # GPU設定
    'is_gpu_conversion': False,
    'is_use_opencl': False,
    
    # 出力設定
    'is_normalization': False,
    'is_add_model_name': False,
    'is_create_model_folder': False,
}
```

## 9. エラーハンドリング

### エラー処理メソッド
- `error_dialoge()`: エラーダイアログ表示
- `message_box()`: メッセージボックス表示
- `process_end(error)`: エラー時の終了処理

### ログ機能
- `command_Text`: ThreadSafeConsole でリアルタイム出力
- `error_log_var`: エラーログ保持
- プロセス進行状況の詳細ログ

## 10. プログレスバー・コンソール出力

### プログレス表示
```python
'progress_bar_main_var': メインプログレスバー (0-100)
'conversion_Button_Text_var': ボタンテキスト更新
- "Start Processing" → "Process Progress: X%" → "Start Processing"
```

### コンソール出力
- **ThreadSafeConsole**: スレッドセーフなコンソール
- **リアルタイム更新**: 処理状況をリアルタイム表示
- **右クリックメニュー**: ログコピー機能

## 11. その他の重要な機能

### ダウンロードセンター
```python
# モデルダウンロード関連変数
'model_download_demucs_var'
'model_download_mdx_var' 
'model_download_vr_var'
'download_progress_bar_var'
'online_data': オンラインモデル情報
```

### キャッシュシステム
- **ソースキャッシュ**: `cached_sources_clear()`
- **モデルキャッシュ**: VR/MDX/Demucs別管理
- **トーチキャッシュ**: `clear_cache_torch`

### 入力検証
- **ファイル形式チェック**: 対応形式の検証
- **パス検証**: 入力/出力パスの有効性確認
- **モデル検証**: 選択モデルの存在確認
- **ストレージチェック**: 十分な空き容量確認

### 国際化対応
- **多言語対応**: constants.pyで文字列定数管理
- **UI文字列**: 統一された文字列管理システム

### パフォーマンス最適化
- **マルチスレッド処理**: KThreadクラス使用
- **メモリ管理**: キャッシュクリア機能
- **GPU最適化**: CUDA/DirectML対応

## 技術仕様

- **GUI フレームワーク**: Tkinter + ttk + sv_ttk (ダークテーマ)
- **音声処理**: PyTorch + ONNX Runtime
- **設定管理**: JSON形式
- **総行数**: 7,265行
- **メソッド数**: 221個（MainWindowクラス）

このレポートは UVR.py の主要機能を網羅的に調査した結果です。
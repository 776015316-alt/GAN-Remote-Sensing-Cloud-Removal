# 数据集与实验结果

数据集和实验结果保存在本仓库的 [Releases 下载区](https://github.com/776015316-alt/GAN-Remote-Sensing-Cloud-Removal/releases/tag/data-results-2026-10-18)，没有放入 Git 文件树。

下载所需的全部 ZIP 分卷后，将每个 ZIP 解压到项目根目录。压缩包保留了原始的 `data/`、`RICE1_results/` 等目录路径；分卷之间没有重复文件。下载区的 `manifest.json` 列出了每个压缩包的文件数、大小和 SHA-256 校验值。

Windows PowerShell 示例（在压缩包所在目录运行，将目标路径改为你的项目目录）：

```powershell
Get-ChildItem -Filter '*.zip' | ForEach-Object {
    Expand-Archive -LiteralPath $_.FullName -DestinationPath 'D:\path\to\Dense_HyA' -Force
}
```

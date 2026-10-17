
:: 启动 Python 预测脚本
python predict.py ^
    --config config.yml ^
    --test_dir C:/Users/Administrator/Desktop ^
    --out_dir C:/Users/Administrator/Desktop/0 ^
    --pretrained RICE1_2_ALLMERGED_results/000034/models/gen_model_epoch_150.pth ^
    --cuda

:: 防止窗口自动关闭
echo 完成预测，按任意键退出...
pause >nul
# 1. 删除已添加的文件
git rm --cached dataset/fer2013.tar.gz
git rm --cached model/EmoNeXt_base.pkl
git commit -m "Remove large files from Git index"

# 2. 安装 Git LFS 并跟踪大文件
git lfs install
git lfs track "dataset/fer2013.tar.gz"
git lfs track "model/EmoNeXt_base.pkl"

# 3. 检查 .gitattributes 文件（可选）
# cat .gitattributes

# 4. 重新添加并提交文件
git add dataset/fer2013.tar.gz
git add model/EmoNeXt_base.pkl
git add .gitattributes
git commit -m "Re-add large files via Git LFS"

# 5. 强制推送到远程仓库
git push origin main --force

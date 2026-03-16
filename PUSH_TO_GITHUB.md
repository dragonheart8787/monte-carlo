# 推送到 GitHub

## 1. 建立 GitHub 儲存庫

1. 前往 https://github.com/new
2. 儲存庫名稱建議：`monte-carlo-comm` 或 `蒙地卡羅通訊`
3. 選擇 Public
4. **不要**勾選 "Add a README"（本地已有）
5. 點擊 Create repository

## 2. 新增遠端並推送

建立儲存庫後，GitHub 會顯示指令。將 `YOUR_USERNAME` 換成你的 GitHub 帳號：

```bash
cd "C:\Users\User\Desktop\蒙地卡羅"
git remote add origin https://github.com/YOUR_USERNAME/monte-carlo-comm.git
git branch -M main
git push -u origin main
```

若使用 SSH：

```bash
git remote add origin git@github.com:YOUR_USERNAME/monte-carlo-comm.git
git branch -M main
git push -u origin main
```

## 3. 已完成

- Git 已初始化
- 已建立 .gitignore
- 已提交所有檔案（Initial commit）
- 僅需建立遠端儲存庫並執行上述 push 指令

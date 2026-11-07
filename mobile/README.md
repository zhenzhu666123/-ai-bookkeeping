# AI 家庭记账 · Mobile

用 **Capacitor** 把 Web 版打包成原生 App，Android / iOS 都能出。

## 目录结构

```
mobile/
├── www/                      ← Web 资源（打包后就是 App 里的静态页面）
│   ├── index.html            ← 从 static/index.html 复制过来，改了 API 地址
│   ├── manifest.json         ← PWA 清单（浏览器添加到主屏用）
│   └── icons/                ← PWA 图标
├── android/                  ← 跑 npx cap add android 后自动生成
├── ios/                      ← 跑 npx cap add ios 后自动生成
├── capacitor.config.js       ← App ID / 名称 / 配置
├── package.json              ← npm 依赖
├── gen_assets.py             ← 图标启动图生成脚本
└── README.md
```

## 前置准备

**必须要装的：**

1. **Node.js 18+** 和 npm  
   https://nodejs.org 下载 LTS 版本  
   验证：`node -v` `npm -v`

2. **Android Studio**（只出 Android 需要这个）  
   https://developer.android.com/studio  
   安装后打开一次让它初始化 SDK

**iOS 需要：**  
macOS + Xcode（没 Mac 就别折腾 iOS）

## 五步打包 Android APK

### 1. 改后端地址（重要！）

打开 `www/index.html`，找到：
```javascript
const API = 'https://你的后端域名.com';
```

改成三种之一：

| 场景 | 填什么 |
|---|---|
| 电脑本地跑后端（同网络）| `http://192.168.1.100:8000`（替换成你电脑的局域网 IP，`ipconfig` 查） |
| Cloudflare 隧道（临时调试用）| `https://xxxx.trycloudflare.com` |
| 云服务器部署 | `https://api.yourdomain.com` |

> ⚠️ **打包后是跨域请求**，后端必须已开启 CORS（本项目 `main.py` 已开）。

### 2. 装依赖

```bash
cd mobile
npm install
```

### 3. 生成图标（可选，但推荐）

```bash
python gen_assets.py
```

会生成蓝底"账"字图标 + 启动页。想换颜色/字，改 `gen_assets.py` 里 `PRIMARY` 和 `text = "账"`。

### 4. 添加 Android 平台

```bash
npx cap add android
```

如果报错找不到 SDK，用 Android Studio 打开一次让它初始化好，再回来跑。

### 5. 同步 & 打包

```bash
npx cap sync          # 把 www/ 同步进 android/
cd android
./gradlew assembleDebug
```

Windows 用户跑 `gradlew.bat assembleDebug`（不带 `./`）。

**APK 位置：**  
`android/app/build/outputs/apk/debug/app-debug.apk`

## 安装到手机

**方法 A（USB 调试）：**  
1. 手机开**开发者模式**（设置 → 关于手机 → 连点 7 次版本号）
2. 开**USB 调试**（开发者选项里）
3. USB 数据线连电脑
4. 跑：
   ```bash
   adb install app-debug.apk
   ```

**方法 B（直接拷贝）：**  
1. 把 `app-debug.apk` 拷到手机存储
2. 手机文件管理器点开
3. 允许"未知来源应用"
4. 安装

**方法 C（Capacitor App Builder 云打包）：**  
不想折腾 Android Studio 用云端打包：  
https://www.app-builder.capacitorjs.com/  
1. 把 `mobile/` 目录 push 到 GitHub
2. 网站关联仓库、选分支、点 Build
3. 下载生成的 APK  
免费额度 1000 次/月。

## 常见问题

**Q: `npx cap add android` 报 SDK 找不到**  
打开 Android Studio 一次，让它下载 SDK，回来再跑。

**Q: `./gradlew` 报 Java 版本错**  
确保装了 JDK 17：`java -version`

**Q: 打包完 App 打开是空白**  
99% 是 `www/index.html` 里的 `const API` 没改，或者后端没开。打开 Android 模拟器看 logcat：
```bash
adb logcat | grep "chromium"
```

**Q: 换了 API 地址 App 里没生效**  
```bash
npx cap sync        # 重新同步
./gradlew assembleDebug
```

## 打包发布到应用商店（可选）

**Google Play：**
1. 申请开发者账号（$25）
2. `./gradlew assembleRelease` 出 release APK
3. 用 keystore 签名：
   ```bash
   keytool -genkey -v -keystore my-release-key.keystore -alias my-key-alias -keyalg RSA -keysize 2048 -validity 10000
   ```
4. Play Console 上传 APK/AAB

**App Store：**
1. 苹果开发者账号（$99/年）
2. `npx cap add ios`
3. `npx cap open ios`（打开 Xcode）
4. Xcode 签名后 Archive → App Store Connect 上传

调试和内部演示**用 debug APK 就够了**，不用上架。

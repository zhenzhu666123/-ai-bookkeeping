/**
 * Capacitor 配置
 *
 * appId:    应用唯一标识符，用于应用商店识别。格式 com.公司.应用名
 * appName:  App 上显示的名字
 * webDir:   Web 项目构建输出目录（我们的静态 HTML 都在 www/ 里）
 *
 * 修改建议：
 * - appId 建议改成 com.你的姓.你的名.bookkeeping（比如 com.zhangsan.bookkeeping）
 * - appName 可以随便改，改后记得重新 npx cap sync
 */
const config = {
  appId: 'com.chenwj.bookkeeping',
  appName: 'AI 家庭记账',
  webDir: 'www',
  backgroundColor: '#ffffff',

  android: {
    // 允许混合内容加载（HTTP API + HTTPS 页面）
    allowMixedContent: true,
  },

  ios: {
    // 自动处理刘海屏、底部安全区
    contentInset: 'automatic',
  },

  server: {
    // 使用本地资源，不走 Android 网络缓存
    androidScheme: 'https',
  },
};

export default config;

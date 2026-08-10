import { useState } from 'react'
import { api } from '../api'
import type { AuthUser } from '../types'

export function AuthPage({ onAuthenticated }: { onAuthenticated: (user: AuthUser) => void }) {
  const [mode, setMode] = useState<'login' | 'register'>('login')
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [confirmPassword, setConfirmPassword] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')

  async function submit() {
    if (!username.trim() || !password) return
    if (mode === 'register' && password !== confirmPassword) {
      setError('两次输入的密码不一致。')
      return
    }
    setBusy(true)
    setError('')
    try {
      const user = mode === 'login'
        ? await api.login(username.trim(), password)
        : await api.register(username.trim(), password)
      onAuthenticated(user)
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : '认证请求失败')
    } finally {
      setBusy(false)
    }
  }

  function fillDemoAccount() {
    setMode('login')
    setUsername('admin')
    setPassword('123456')
    setConfirmPassword('')
    setError('')
  }

  return (
    <main className="auth-shell">
      <section className="auth-panel" aria-labelledby="auth-title">
        <header>
          <small>IDENTITY ACCESS · YH CONTROL</small>
          <h1 id="auth-title">进入曜衡智控平台</h1>
          <p>登录后可使用诊断、安防处置和个人对话历史。</p>
        </header>
        <div className="auth-tabs" role="tablist" aria-label="账号操作">
          <button type="button" role="tab" aria-selected={mode === 'login'} onClick={() => { setMode('login'); setError('') }}>登录</button>
          <button type="button" role="tab" aria-selected={mode === 'register'} onClick={() => { setMode('register'); setError('') }}>注册</button>
        </div>
        <div className="auth-form">
          <label><span>用户名</span><input autoComplete="username" value={username} onChange={(event) => setUsername(event.target.value)} placeholder="3–32 位中英文、数字或 _ -" /></label>
          <label><span>密码</span><input type="password" autoComplete={mode === 'login' ? 'current-password' : 'new-password'} value={password} onChange={(event) => setPassword(event.target.value)} placeholder="至少 6 位" onKeyDown={(event) => { if (event.key === 'Enter') void submit() }} /></label>
          {mode === 'register' && <label><span>确认密码</span><input type="password" autoComplete="new-password" value={confirmPassword} onChange={(event) => setConfirmPassword(event.target.value)} onKeyDown={(event) => { if (event.key === 'Enter') void submit() }} /></label>}
          {error && <div className="auth-error" role="alert">{error}</div>}
          <button className="button button-primary auth-submit" type="button" disabled={busy || !username.trim() || !password || (mode === 'register' && !confirmPassword)} onClick={() => void submit()}>{busy ? '正在验证…' : mode === 'login' ? '登录平台' : '创建账号并登录'}</button>
          {mode === 'login' && <button className="auth-demo" type="button" onClick={fillDemoAccount}><span>DEMO</span>填入演示账号 admin / 123456</button>}
        </div>
      </section>
    </main>
  )
}

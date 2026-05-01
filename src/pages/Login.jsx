import React, { useState } from 'react';
import { useNavigate, Link, useLocation } from 'react-router-dom';
import { signInWithEmailAndPassword } from 'firebase/auth';
import { auth } from '../firebase';
import { Button } from '../components/ui/Button';
import { Input } from '../components/ui/Input';
import { Sparkles, ArrowRight, Loader2, Mail, Lock, ShieldCheck } from 'lucide-react';

export const Login = () => {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  const navigate = useNavigate();
  const location = useLocation();
  const from = location.state?.from?.pathname || '/dashboard';

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    setLoading(true);
    try {
      await signInWithEmailAndPassword(auth, email, password);
      navigate(from, { replace: true });
    } catch (err) {
      console.error("Auth Error:", err.code);
      let msg = "Authentication Failed.";
      
      switch (err.code) {
        case 'auth/user-not-found':
          msg = "No account found with this email. Please signup first.";
          break;
        case 'auth/wrong-password':
          msg = "Incorrect access key (password). Please try again.";
          break;
        case 'auth/invalid-email':
          msg = "The identity (email) format is invalid.";
          break;
        case 'auth/operation-not-allowed':
          msg = "Email/Password sign-in is not enabled in Firebase Console.";
          break;
        case 'auth/network-request-failed':
          msg = "Network error. Check your connection or Firebase authorized domains.";
          break;
        default:
          msg = err.message.replace('Firebase: ', '');
      }
      setError(msg);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-[var(--color-background)] p-6 relative overflow-hidden">
      {/* Background Decorations */}
      <div className="absolute top-0 right-0 -mr-20 -mt-20 w-96 h-96 bg-primary/5 rounded-full blur-[100px]" />
      <div className="absolute bottom-0 left-0 -ml-20 -mb-20 w-96 h-96 bg-primary/10 rounded-full blur-[100px]" />
      
      <div className="w-full max-w-xl relative">
        <div className="text-center mb-12">
          <div className="inline-flex items-center justify-center w-20 h-20 bg-primary rounded-[2rem] text-white shadow-2xl shadow-primary/30 mb-8 animate-bounce">
            <Sparkles className="w-10 h-10" />
          </div>
          <h1 className="text-6xl font-black text-slate-800 tracking-tighter mb-4">
            Welcome <span className="text-primary italic">Back.</span>
          </h1>
          <p className="text-xl text-slate-500 font-medium tracking-tight">
            Continue your journey to technical mastery.
          </p>
        </div>

        <div className="glass-card rounded-[3.5rem] p-12 bg-white/40 border-primary/5 shadow-2xl">
          <form onSubmit={handleSubmit} className="space-y-8">
            {error && (
              <div className="rounded-2xl bg-red-50 border border-red-100 p-4 text-sm font-bold text-red-700 flex items-center gap-3">
                <ShieldCheck className="w-5 h-5 shrink-0" />
                {error}
              </div>
            )}

            <div className="space-y-3">
              <label className="text-sm font-black text-slate-700 uppercase tracking-widest ml-1">Identity (Email)</label>
              <div className="relative">
                <Mail className="absolute left-6 top-1/2 -translate-y-1/2 text-slate-300 w-5 h-5" />
                <input 
                  type="email" 
                  placeholder="name@nexus.com" 
                  value={email}
                  onChange={(e) => setEmail(e.target.value)} 
                  required 
                  className="w-full bg-white/50 border-2 border-primary/5 rounded-[1.5rem] py-5 pl-16 pr-6 font-bold text-slate-700 outline-none focus:border-primary/20 transition-all shadow-sm"
                />
              </div>
            </div>

            <div className="space-y-3">
              <label className="text-sm font-black text-slate-700 uppercase tracking-widest ml-1">Access Key (Password)</label>
              <div className="relative">
                <Lock className="absolute left-6 top-1/2 -translate-y-1/2 text-slate-300 w-5 h-5" />
                <input 
                  type="password" 
                  value={password}
                  onChange={(e) => setPassword(e.target.value)} 
                  required 
                  placeholder="••••••••"
                  className="w-full bg-white/50 border-2 border-primary/5 rounded-[1.5rem] py-5 pl-16 pr-6 font-bold text-slate-700 outline-none focus:border-primary/20 transition-all shadow-sm"
                />
              </div>
            </div>

            <Button 
              className="w-full btn-primary rounded-[1.5rem] py-8 text-xl font-black shadow-2xl shadow-primary/20 hover:scale-[1.02] active:scale-95 transition-all" 
              type="submit" 
              disabled={loading}
            >
              {loading ? (
                <Loader2 className="w-6 h-6 animate-spin" />
              ) : (
                <span className="flex items-center gap-3">
                  Authenticate <ArrowRight className="w-6 h-6" />
                </span>
              )}
            </Button>
          </form>

          <div className="mt-10 text-center">
            <p className="text-slate-500 font-bold">
              New to the platform?{' '}
              <Link to="/signup" className="text-primary underline decoration-emerald-100 decoration-4 underline-offset-4 hover:text-primary-dark transition-colors">
                Initialize Account
              </Link>
            </p>
          </div>
        </div>

        {/* Footer info */}
        <p className="text-center mt-12 text-slate-400 text-xs font-black uppercase tracking-[0.2em]">
          Secured by LearnPath Neural Auth Engine v4.2
        </p>
      </div>
    </div>
  );
};

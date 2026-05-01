import React, { useState } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import { createUserWithEmailAndPassword, updateProfile } from 'firebase/auth';
import { auth } from '../firebase';
import { Button } from '../components/ui/Button';
import { Sparkles, ArrowRight, Loader2, User, Mail, Lock, CheckCircle2 } from 'lucide-react';

export const Signup = () => {
  const [name, setName] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [success, setSuccess] = useState(false);
  const [loading, setLoading] = useState(false);

  const navigate = useNavigate();

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    setLoading(true);
    try {
      const userCredential = await createUserWithEmailAndPassword(auth, email, password);
      await updateProfile(userCredential.user, { displayName: name });
      setSuccess(true);
      setTimeout(() => navigate('/dashboard'), 2000);
    } catch (err) {
      console.error("Signup Error:", err.code);
      let msg = "Failed to create account.";
      
      switch (err.code) {
        case 'auth/email-already-in-use':
          msg = "This identity (email) is already linked to an account.";
          break;
        case 'auth/weak-password':
          msg = "The access key (password) is too weak. Use at least 6 characters.";
          break;
        case 'auth/invalid-email':
          msg = "The identity (email) format is invalid.";
          break;
        case 'auth/operation-not-allowed':
          msg = "Signup is not enabled in Firebase Console.";
          break;
        default:
          msg = err.message.replace('Firebase: ', '');
      }
      setError(msg);
    } finally {
      setLoading(false);
    }
  };

  if (success) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-[var(--color-background)]">
        <div className="text-center animate-in zoom-in duration-500">
          <div className="w-24 h-24 bg-primary text-white rounded-full flex items-center justify-center mx-auto mb-8 shadow-2xl shadow-primary/30">
            <CheckCircle2 className="w-12 h-12" />
          </div>
          <h2 className="text-5xl font-black text-slate-800 tracking-tighter mb-4">Neural Link Established!</h2>
          <p className="text-xl text-slate-500 font-bold">Welcome to the future of learning, {name}.</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen flex items-center justify-center bg-[var(--color-background)] p-6 relative overflow-hidden">
      <div className="absolute top-0 left-0 -ml-20 -mt-20 w-96 h-96 bg-primary/5 rounded-full blur-[100px]" />
      <div className="absolute bottom-0 right-0 -mr-20 -mb-20 w-96 h-96 bg-primary/10 rounded-full blur-[100px]" />

      <div className="w-full max-w-xl relative">
        <div className="text-center mb-12">
          <div className="inline-flex items-center justify-center w-20 h-20 bg-primary rounded-[2rem] text-white shadow-2xl shadow-primary/30 mb-8">
            <Sparkles className="w-10 h-10" />
          </div>
          <h1 className="text-6xl font-black text-slate-800 tracking-tighter mb-4">
            Join the <span className="text-primary italic">Nexus.</span>
          </h1>
          <p className="text-xl text-slate-500 font-medium tracking-tight">
            Start your bespoke 100-step trajectory today.
          </p>
        </div>

        <div className="glass-card rounded-[3.5rem] p-12 bg-white/40 border-primary/5 shadow-2xl">
          <form onSubmit={handleSubmit} className="space-y-6">
            {error && (
              <div className="rounded-2xl bg-red-50 border border-red-100 p-4 text-sm font-bold text-red-700">
                {error}
              </div>
            )}

            <div className="space-y-2">
              <label className="text-sm font-black text-slate-700 uppercase tracking-widest ml-1">Full Name</label>
              <div className="relative">
                <User className="absolute left-6 top-1/2 -translate-y-1/2 text-slate-300 w-5 h-5" />
                <input 
                  type="text" 
                  placeholder="John Architect" 
                  value={name}
                  onChange={(e) => setName(e.target.value)} 
                  required 
                  className="w-full bg-white/50 border-2 border-primary/5 rounded-[1.5rem] py-5 pl-16 pr-6 font-bold text-slate-700 outline-none focus:border-primary/20 transition-all"
                />
              </div>
            </div>

            <div className="space-y-2">
              <label className="text-sm font-black text-slate-700 uppercase tracking-widest ml-1">Email Address</label>
              <div className="relative">
                <Mail className="absolute left-6 top-1/2 -translate-y-1/2 text-slate-300 w-5 h-5" />
                <input 
                  type="email" 
                  placeholder="john@nexus.com" 
                  value={email}
                  onChange={(e) => setEmail(e.target.value)} 
                  required 
                  className="w-full bg-white/50 border-2 border-primary/5 rounded-[1.5rem] py-5 pl-16 pr-6 font-bold text-slate-700 outline-none focus:border-primary/20 transition-all"
                />
              </div>
            </div>

            <div className="space-y-2">
              <label className="text-sm font-black text-slate-700 uppercase tracking-widest ml-1">Create Access Key</label>
              <div className="relative">
                <Lock className="absolute left-6 top-1/2 -translate-y-1/2 text-slate-300 w-5 h-5" />
                <input 
                  type="password" 
                  value={password}
                  onChange={(e) => setPassword(e.target.value)} 
                  required 
                  placeholder="Minimum 8 characters"
                  className="w-full bg-white/50 border-2 border-primary/5 rounded-[1.5rem] py-5 pl-16 pr-6 font-bold text-slate-700 outline-none focus:border-primary/20 transition-all"
                />
              </div>
            </div>

            <Button 
              className="w-full btn-primary rounded-[1.5rem] py-8 text-xl font-black shadow-2xl shadow-primary/20 hover:scale-[1.02] active:scale-95 transition-all mt-4" 
              type="submit" 
              disabled={loading}
            >
              {loading ? (
                <Loader2 className="w-6 h-6 animate-spin" />
              ) : (
                <span className="flex items-center gap-3">
                  Begin Trajectory <ArrowRight className="w-6 h-6" />
                </span>
              )}
            </Button>
          </form>

          <div className="mt-8 text-center text-slate-500 font-bold">
            Already a member?{' '}
            <Link to="/login" className="text-primary underline decoration-emerald-100 decoration-4 underline-offset-4 hover:text-primary-dark">
              Authenticate Now
            </Link>
          </div>
        </div>
      </div>
    </div>
  );
};

import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { 
  Search, 
  ArrowRight, 
  Sparkles, 
  Zap, 
  Layers, 
  Shield, 
  Cpu, 
  Gamepad, 
  Database,
  Cloud,
  Palette,
  Code2
} from 'lucide-react';
import { Button } from '../components/ui/Button';
import api from '../services/api';

const domains = [
  { id: 'web', name: 'Web Dev', icon: Layers, color: 'bg-emerald-50 text-primary' },
  { id: 'blockchain', name: 'Blockchain', icon: Cpu, color: 'bg-emerald-50 text-primary' },
  { id: 'cyber', name: 'Cybersecurity', icon: Shield, color: 'bg-emerald-50 text-primary' },
  { id: 'cloud', name: 'Cloud Computing', icon: Cloud, color: 'bg-emerald-50 text-primary' },
  { id: 'game', name: 'Game Dev', icon: Gamepad, color: 'bg-emerald-50 text-primary' },
  { id: 'data', name: 'Data Science', icon: Database, color: 'bg-emerald-50 text-primary' },
  { id: 'design', name: 'UI/UX Design', icon: Palette, color: 'bg-emerald-50 text-primary' },
  { id: 'devops', name: 'DevOps', icon: Code2, color: 'bg-emerald-50 text-primary' },
];

export const Recommendations = () => {
  const navigate = useNavigate();
  const [interest, setInterest] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  const handleGenerate = async (selectedInterest) => {
    const targetInterest = selectedInterest || interest;
    if (!targetInterest) return;

    // Navigate immediately to show the beautiful architecting screen on the next page
    navigate('/learning-path', { state: { interest: targetInterest } });
  };

  return (
    <div className="space-y-16 pb-20">
      {/* Search Section */}
      <section className="text-center max-w-4xl mx-auto pt-10">
        <div className="inline-flex items-center gap-3 px-6 py-2 rounded-full bg-primary/10 text-primary mb-8 animate-bounce">
          <Sparkles className="w-5 h-5" />
          <span className="text-sm font-black uppercase tracking-widest">AI-Powered Discovery</span>
        </div>
        <h1 className="text-7xl font-black text-slate-800 mb-8 tracking-tighter leading-none">
          What do you want <br/>
          <span className="text-primary underline decoration-emerald-100 decoration-8 underline-offset-8 italic">to master next?</span>
        </h1>
        <p className="text-2xl text-slate-500 font-medium mb-12 leading-relaxed">
          Tell our AI your goal, and we'll craft a bespoke 100-step roadmap <br/> 
          architected specifically for your current skill level.
        </p>

        <div className="relative group max-w-2xl mx-auto">
          <div className="absolute -inset-2 bg-primary/20 rounded-[2.5rem] blur-xl opacity-0 group-hover:opacity-100 transition-opacity duration-500" />
          <div className="relative flex items-center bg-white p-3 rounded-[2.5rem] shadow-2xl border border-primary/10">
            <Search className="ml-6 w-8 h-8 text-slate-300" />
            <input
              type="text"
              placeholder="e.g. Master React and Next.js..."
              className="flex-1 bg-transparent border-none outline-none px-6 text-xl font-bold text-slate-700 placeholder:text-slate-300"
              value={interest}
              onChange={(e) => setInterest(e.target.value)}
              onKeyPress={(e) => e.key === 'Enter' && handleGenerate()}
            />
            <Button 
              onClick={() => handleGenerate()}
              disabled={isLoading || !interest}
              className="btn-primary rounded-[2rem] px-10 py-5 text-lg font-black h-auto"
            >
              {isLoading ? 'Architecting...' : 'Generate Path'}
            </Button>
          </div>
        </div>
      </section>

      {/* Domain Grid */}
      <section className="space-y-10">
        <div className="flex items-center justify-between">
          <h2 className="text-4xl font-black text-slate-800 tracking-tight">Trending Domains</h2>
          <div className="h-px flex-1 mx-10 bg-gradient-to-r from-primary/20 to-transparent" />
        </div>

        <div className="grid grid-cols-2 md:grid-cols-4 gap-8">
          {domains.map((domain) => (
            <button
              key={domain.id}
              onClick={() => handleGenerate(domain.name)}
              className="glass-card group p-10 rounded-[3rem] text-center hover:scale-105 hover:shadow-[0_30px_60px_-15px_rgba(0,161,155,0.2)] transition-all duration-500 border-2 border-transparent hover:border-primary/20"
            >
              <div className={`w-24 h-24 mx-auto rounded-[2rem] flex items-center justify-center mb-8 transition-all duration-500 group-hover:rotate-6 ${domain.color} shadow-sm group-hover:shadow-lg`}>
                <domain.icon className="w-12 h-12" />
              </div>
              <h3 className="text-xl font-black text-slate-800 group-hover:text-primary transition-colors">{domain.name}</h3>
              <div className="mt-4 flex items-center justify-center gap-2 text-primary opacity-0 group-hover:opacity-100 transition-all">
                <span className="text-xs font-black uppercase tracking-tighter">Generate Roadmap</span>
                <ArrowRight className="w-4 h-4" />
              </div>
            </button>
          ))}
        </div>
      </section>

      {/* Featured Insight */}
      <section className="glass-card rounded-[4rem] p-16 flex flex-col md:flex-row items-center gap-16 bg-white/40 border-primary/5">
        <div className="w-40 h-40 bg-primary rounded-full flex items-center justify-center shrink-0 shadow-2xl shadow-primary/20">
          <Zap className="w-20 h-20 text-white" />
        </div>
        <div>
          <h2 className="text-4xl font-black text-slate-800 mb-6">Hyper-Personalized Learning</h2>
          <p className="text-xl text-slate-500 font-medium leading-relaxed">
            Our neural engine analyzes thousands of successful career trajectories to recommend paths that lead directly to professional mastery. No fluff, just the steps that matter.
          </p>
        </div>
      </section>
    </div>
  );
};

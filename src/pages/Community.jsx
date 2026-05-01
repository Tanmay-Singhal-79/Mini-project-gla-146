import React, { useState } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { motion, AnimatePresence } from 'framer-motion';
import { ThumbsUp, MessageSquare, ExternalLink, Plus, Share2, Users, Sparkles, Filter, ChevronDown } from 'lucide-react';
import api from '../services/api';
import { Button } from '../components/ui/Button';
import { CommentSection } from '../components/community/CommentSection';

export const Community = () => {
  const [newResource, setNewResource] = useState('');
  const [expandedResource, setExpandedResource] = useState(null);
  const queryClient = useQueryClient();

  const { data, isLoading } = useQuery({
    queryKey: ['resources'],
    queryFn: async () => {
      // Short delay for "Sync" feel
      await new Promise(resolve => setTimeout(resolve, 800));
      try {
        const res = await api.community.getResources();
        if (res.data && res.data.length > 0) return res.data;
        throw new Error("Empty Community");
      } catch (err) {
        console.warn("Community sync failed, using Social Cache:", err);
        const { mockResources } = await import('../data/mockCommunity');
        return mockResources;
      }
    },
  });

  const addMutation = useMutation({
    mutationFn: (title) => api.community.addResource(title),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['resources'] });
      setNewResource('');
    }
  });

  const upvoteMutation = useMutation({
    mutationFn: (id) => api.community.upvote(id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['resources'] });
    }
  });

  const handleAdd = (e) => {
    e.preventDefault();
    if (newResource.trim()) {
      addMutation.mutate(newResource.trim());
    }
  };

  return (
    <div className="space-y-16 pb-32">
      {/* Header Section */}
      <section className="flex flex-col md:flex-row md:items-end justify-between gap-12 pt-10">
        <div className="max-w-2xl">
          <motion.div 
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            className="inline-flex items-center gap-3 px-5 py-2 rounded-full bg-primary/10 text-primary mb-8"
          >
            <Users className="w-5 h-5" />
            <span className="text-[10px] font-black uppercase tracking-[0.3em]">Neural Network Hub</span>
          </motion.div>
          <h1 className="text-7xl font-black text-slate-800 tracking-tighter mb-8 leading-[0.9]">
            Collective <br/>
            <span className="text-primary italic underline decoration-emerald-100 decoration-8 underline-offset-8">Intelligence</span>
          </h1>
          <p className="text-2xl text-slate-500 font-medium leading-relaxed">
            Architect the future with thousands of developers. Share high-signal patterns, discuss logic, and evolve together.
          </p>
        </div>
        
        <div className="flex items-center gap-6">
          <Button variant="outline" className="rounded-2xl px-8 py-8 border-primary/10 text-primary font-black hover:bg-primary/5">
            <Filter className="w-5 h-5 mr-3" /> Filter Signals
          </Button>
          <Button className="btn-primary rounded-[2rem] px-10 py-8 font-black shadow-2xl shadow-primary/30">
            <Share2 className="w-5 h-5 mr-3" /> Sync Friend
          </Button>
        </div>
      </section>

      {/* Share Box */}
      <motion.section 
        initial={{ opacity: 0, y: 30 }}
        animate={{ opacity: 1, y: 0 }}
        className="glass-card rounded-[4rem] p-12 bg-white/60 relative overflow-hidden group"
      >
        <div className="absolute -left-20 -top-20 w-64 h-64 bg-primary/5 rounded-full blur-3xl group-focus-within:scale-150 transition-transform duration-1000" />
        <form onSubmit={handleAdd} className="relative z-10 flex flex-col md:flex-row gap-8 items-center">
          <div className="flex-1 w-full relative">
            <div className="absolute left-8 top-1/2 -translate-y-1/2 text-primary/40">
              <Plus className="w-8 h-8" />
            </div>
            <input 
              value={newResource}
              onChange={(e) => setNewResource(e.target.value)}
              placeholder="Deploy a masterpiece URL or architectural resource..." 
              className="w-full bg-white border-2 border-primary/5 rounded-[2.5rem] py-7 pl-20 pr-8 text-xl font-bold text-slate-700 placeholder:text-slate-300 focus:border-primary/20 outline-none transition-all shadow-premium"
            />
          </div>
          <Button 
            type="submit" 
            disabled={addMutation.isPending || !newResource.trim()}
            className="btn-primary rounded-[2.5rem] px-16 py-7 text-xl font-black h-auto w-full md:w-auto shadow-2xl"
          >
            {addMutation.isPending ? 'Syncing...' : 'Deploy Node'}
          </Button>
        </form>
      </motion.section>

      {/* Resources Grid */}
      <section className="grid gap-10">
        {isLoading ? (
          [1, 2, 3].map(i => (
            <div key={i} className="h-40 glass-card rounded-[3rem] animate-pulse" />
          ))
        ) : (
          data?.map((resource, index) => (
            <motion.div 
              key={resource.id} 
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: index * 0.1 }}
              className={`glass-card p-10 rounded-[3.5rem] border-2 transition-all duration-500 overflow-hidden ${
                expandedResource === resource.id ? 'border-primary/20' : 'border-transparent hover:border-primary/10'
              }`}
            >
              <div className="flex flex-col md:flex-row items-center justify-between gap-10">
                <div className="flex items-center gap-10 flex-1">
                  <div className="w-20 h-20 shrink-0 bg-emerald-50 rounded-[2rem] flex items-center justify-center text-primary group-hover:scale-110 group-hover:rotate-6 transition-all duration-700 shadow-lg shadow-emerald-900/5">
                    <Sparkles className="w-10 h-10" />
                  </div>
                  <div>
                    <h3 className="text-3xl font-black text-slate-800 mb-4 flex items-center gap-4">
                      {resource.title}
                      <a href="#" className="p-2 hover:bg-primary/5 rounded-xl transition-colors">
                        <ExternalLink className="w-6 h-6 text-slate-300 hover:text-primary" />
                      </a>
                    </h3>
                    <div className="flex flex-wrap items-center gap-8 text-xs font-black text-slate-400 uppercase tracking-widest">
                      <button 
                        onClick={() => setExpandedResource(expandedResource === resource.id ? null : resource.id)}
                        className="flex items-center gap-3 hover:text-primary transition-colors bg-primary/5 px-4 py-2 rounded-full"
                      >
                        <MessageSquare className="w-5 h-5" /> 
                        {resource.comments_count || 0} Discussions
                        <ChevronDown className={`w-4 h-4 transition-transform ${expandedResource === resource.id ? 'rotate-180' : ''}`} />
                      </button>
                      <span className="flex items-center gap-3 px-4 py-2 border border-slate-100 rounded-full">
                        <Users className="w-5 h-5" /> {resource.creator_name || 'Architect'}
                      </span>
                    </div>
                  </div>
                </div>
                
                <div className="flex items-center gap-6 bg-emerald-50/50 p-6 rounded-[2.5rem] border border-primary/10 shadow-inner">
                  <span className="text-4xl font-black text-primary px-4 min-w-[4rem] text-center">
                    {resource.upvotes}
                  </span>
                  <Button 
                    onClick={() => upvoteMutation.mutate(resource.id)}
                    className="w-16 h-16 rounded-full bg-white border-2 border-primary/20 text-primary hover:bg-primary hover:text-white transition-all shadow-xl hover:scale-110 active:scale-90"
                  >
                    <ThumbsUp className="w-7 h-7" />
                  </Button>
                </div>
              </div>

              <AnimatePresence>
                {expandedResource === resource.id && (
                  <CommentSection resourceId={resource.id} />
                )}
              </AnimatePresence>
            </motion.div>
          ))
        )}
      </section>
    </div>
  );
};

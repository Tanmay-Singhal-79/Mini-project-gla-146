import React, { createContext, useContext, useState, useEffect } from 'react';
import { onAuthStateChanged, signOut as firebaseSignOut } from 'firebase/auth';
import { auth } from '../firebase';
import { setupAxiosInterceptors } from '../services/api';

const AuthContext = createContext();

export const useAuth = () => useContext(AuthContext);

export const AuthProvider = ({ children }) => {
  const [user, setUser] = useState(null);
  const [token, setToken] = useState(null);
  const [isLoading, setIsLoading] = useState(true);

  const logout = async () => {
    try {
      await firebaseSignOut(auth);
      setToken(null);
      setUser(null);
      localStorage.removeItem('token');
    } catch (error) {
      console.error("Logout error", error);
    }
  };

  useEffect(() => {
    setupAxiosInterceptors(logout);

    // Safety net: never hang on loading screen for more than 10 seconds
    const loadingTimeout = setTimeout(() => setIsLoading(false), 10000);

    const unsubscribe = onAuthStateChanged(auth, async (currentUser) => {
      clearTimeout(loadingTimeout);
      if (currentUser) {
        // Use cached token on load (fast, no network). getIdToken() auto-refreshes
        // if the cached token is already expired — but does NOT force a network call.
        const idToken = await currentUser.getIdToken();
        setToken(idToken);
        setUser({ 
          uid: currentUser.uid,
          email: currentUser.email,
          name: currentUser.displayName || 'User'
        });
        localStorage.setItem('token', idToken);

        // ─── Proactively refresh every 55 min (tokens expire after 1 hour) ────
        const refreshInterval = setInterval(async () => {
          try {
            // Force refresh here (in background, not blocking UI)
            const newToken = await currentUser.getIdToken(true);
            setToken(newToken);
            localStorage.setItem('token', newToken);
          } catch (err) {
            console.error('Token refresh failed, logging out:', err);
            logout();
          }
        }, 55 * 60 * 1000);

        setIsLoading(false);
        return () => clearInterval(refreshInterval);
      } else {
        setToken(null);
        setUser(null);
        localStorage.removeItem('token');
        setIsLoading(false);
      }
    });

    return () => {
      clearTimeout(loadingTimeout);
      unsubscribe();
    };
  }, []);

  return (
    <AuthContext.Provider value={{ user, token, isLoading, logout }}>
      {children}
    </AuthContext.Provider>
  );
};

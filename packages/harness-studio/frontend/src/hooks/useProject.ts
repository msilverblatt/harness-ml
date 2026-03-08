import { createContext, useContext } from 'react';

export const ProjectContext = createContext<string>('');

/**
 * Returns the current project name from ProjectContext.
 * Must be used within a ProjectContext.Provider (set up in Layout).
 */
export function useProject(): string {
    return useContext(ProjectContext);
}

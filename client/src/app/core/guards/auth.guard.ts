import { inject } from '@angular/core';
import { CanActivateFn, Router } from '@angular/router';
import { AuthService } from '../services/auth.service';

export const authGuard: CanActivateFn = async () => {
  const auth = inject(AuthService);
  const router = inject(Router);

  // If we don't have status yet, check with server
  if (!auth.status()) {
    try {
      await auth.checkStatus();
    } catch {
      router.navigate(['/login']);
      return false;
    }
  }

  const status = auth.status();
  if (!status) {
    router.navigate(['/login']);
    return false;
  }

  // Already authenticated — allow access
  if (status.authenticated) {
    return true;
  }

  // Not authenticated — redirect to login
  router.navigate(['/login']);
  return false;
};

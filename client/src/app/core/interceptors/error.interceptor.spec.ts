import { TestBed } from '@angular/core/testing';
import {
  HttpRequest,
  HttpHandlerFn,
  HttpErrorResponse,
} from '@angular/common/http';
import { throwError } from 'rxjs';
import { errorInterceptor } from './error.interceptor';
import { AuthService } from '../services/auth.service';

describe('errorInterceptor', () => {
  let authMock: { token: string | null; logout: jest.Mock };
  let next: jest.MockedFunction<HttpHandlerFn>;

  beforeEach(() => {
    authMock = { token: null, logout: jest.fn() };
    next = jest.fn();

    TestBed.configureTestingModule({
      providers: [{ provide: AuthService, useValue: authMock }],
    });
  });

  const runInterceptor = (req: HttpRequest<unknown>) =>
    TestBed.runInInjectionContext(() => errorInterceptor(req, next));

  it('calls auth.logout() on 401 for non-auth URLs', (done) => {
    const req = new HttpRequest('GET', '/api/photos');
    const error = new HttpErrorResponse({ status: 401, url: '/api/photos' });
    next.mockReturnValue(throwError(() => error));

    runInterceptor(req).subscribe({
      error: () => {
        expect(authMock.logout).toHaveBeenCalled();
        done();
      },
    });
  });

  it('does NOT call auth.logout() on 401 for /api/auth/ URLs', (done) => {
    const req = new HttpRequest('GET', '/api/auth/status');
    const error = new HttpErrorResponse({ status: 401, url: '/api/auth/status' });
    next.mockReturnValue(throwError(() => error));

    runInterceptor(req).subscribe({
      error: () => {
        expect(authMock.logout).not.toHaveBeenCalled();
        done();
      },
    });
  });

  it('does NOT call auth.logout() on other error codes (404, 500)', (done) => {
    const req = new HttpRequest('GET', '/api/photos');
    const error404 = new HttpErrorResponse({ status: 404, url: '/api/photos' });
    next.mockReturnValue(throwError(() => error404));

    runInterceptor(req).subscribe({
      error: () => {
        expect(authMock.logout).not.toHaveBeenCalled();

        const error500 = new HttpErrorResponse({ status: 500, url: '/api/photos' });
        next.mockReturnValue(throwError(() => error500));

        runInterceptor(req).subscribe({
          error: () => {
            expect(authMock.logout).not.toHaveBeenCalled();
            done();
          },
        });
      },
    });
  });

  it('re-throws the error', (done) => {
    const req = new HttpRequest('GET', '/api/photos');
    const error = new HttpErrorResponse({ status: 401, url: '/api/photos' });
    next.mockReturnValue(throwError(() => error));

    runInterceptor(req).subscribe({
      next: () => {
        done.fail('expected an error');
      },
      error: (err: HttpErrorResponse) => {
        expect(err.status).toBe(401);
        done();
      },
    });
  });
});

import { TestBed, ComponentFixture } from '@angular/core/testing';
import { NO_ERRORS_SCHEMA } from '@angular/core';
import { ActivatedRoute } from '@angular/router';
import { of, throwError } from 'rxjs';
import { ApiService } from '../../core/services/api.service';
import { I18nService } from '../../core/services/i18n.service';
import { PersonPageComponent } from './person-page.component';

// Mock IntersectionObserver which is not available in jsdom
const mockObserve = jest.fn();
const mockDisconnect = jest.fn();
(globalThis as any).IntersectionObserver = jest.fn(() => ({
  observe: mockObserve,
  disconnect: mockDisconnect,
  unobserve: jest.fn(),
}));

describe('PersonPageComponent', () => {
  let fixture: ComponentFixture<PersonPageComponent>;
  let component: PersonPageComponent;
  let mockApi: {
    get: jest.Mock;
    thumbnailUrl: jest.Mock;
    personThumbnailUrl: jest.Mock;
  };
  let mockI18n: { t: jest.Mock };

  const mockPhotosResponse = {
    photos: [
      { path: '/a.jpg', filename: 'a.jpg', aggregate: 8.5, aesthetic: 7.0, date_taken: '2024-01-01' },
      { path: '/b.jpg', filename: 'b.jpg', aggregate: 7.2, aesthetic: 6.5, date_taken: '2024-01-02' },
    ],
    total: 5,
    person: { id: 42, name: 'Alice', face_count: 10 },
  };

  beforeEach(() => {
    mockApi = {
      get: jest.fn(() => of(mockPhotosResponse)),
      thumbnailUrl: jest.fn((path: string) => `/thumb/${path}`),
      personThumbnailUrl: jest.fn((id: number) => `/person_thumbnail/${id}`),
    };
    mockI18n = { t: jest.fn((key: string) => key) };

    TestBed.configureTestingModule({
      imports: [PersonPageComponent],
      providers: [
        { provide: ApiService, useValue: mockApi },
        { provide: I18nService, useValue: mockI18n },
        { provide: ActivatedRoute, useValue: {} },
      ],
      schemas: [NO_ERRORS_SCHEMA],
    });

    fixture = TestBed.createComponent(PersonPageComponent);
    fixture.componentRef.setInput('personId', '42');
    component = fixture.componentInstance;
  });

  describe('initial state', () => {
    it('should have empty photos array', () => {
      expect(component.photos()).toEqual([]);
    });

    it('should have loading as false', () => {
      expect(component.loading()).toBe(false);
    });

    it('should have total as 0', () => {
      expect(component.total()).toBe(0);
    });

    it('should have person as null', () => {
      expect(component.person()).toBeNull();
    });
  });

  describe('hasMore computed', () => {
    it('should return false when photos and total are both 0', () => {
      expect(component.hasMore()).toBe(false);
    });

    it('should return true when photos length is less than total', () => {
      component.photos.set([
        { path: '/a.jpg', filename: 'a.jpg', aggregate: 8, aesthetic: 7, date_taken: '' },
      ]);
      component.total.set(5);
      expect(component.hasMore()).toBe(true);
    });

    it('should return false when photos length equals total', () => {
      component.photos.set([
        { path: '/a.jpg', filename: 'a.jpg', aggregate: 8, aesthetic: 7, date_taken: '' },
        { path: '/b.jpg', filename: 'b.jpg', aggregate: 7, aesthetic: 6, date_taken: '' },
      ]);
      component.total.set(2);
      expect(component.hasMore()).toBe(false);
    });
  });

  describe('ngOnInit', () => {
    it('should call loadPage and set photos from API response', async () => {
      await component.ngOnInit();

      expect(mockApi.get).toHaveBeenCalledWith('/persons/42/photos', {
        page: 1,
        per_page: 48,
      });
      expect(component.photos()).toEqual(mockPhotosResponse.photos);
      expect(component.person()).toEqual(mockPhotosResponse.person);
      expect(component.total()).toBe(5);
    });

    it('should set loading to false after completion', async () => {
      await component.ngOnInit();

      expect(component.loading()).toBe(false);
    });
  });

  describe('loadPage behavior', () => {
    it('should append photos on subsequent calls', async () => {
      const firstPage = {
        photos: [{ path: '/a.jpg', filename: 'a.jpg', aggregate: 8, aesthetic: 7, date_taken: '' }],
        total: 3,
        person: { id: 42, name: 'Alice', face_count: 10 },
      };
      const secondPage = {
        photos: [{ path: '/b.jpg', filename: 'b.jpg', aggregate: 7, aesthetic: 6, date_taken: '' }],
        total: 3,
        person: { id: 42, name: 'Alice', face_count: 10 },
      };

      mockApi.get.mockReturnValueOnce(of(firstPage)).mockReturnValueOnce(of(secondPage));

      await component.ngOnInit();
      expect(component.photos()).toHaveLength(1);
      expect(mockApi.get).toHaveBeenCalledWith('/persons/42/photos', { page: 1, per_page: 48 });

      // Manually invoke loadPage via ngOnInit won't work again, so we access it indirectly.
      // loadPage is private, but after first call page increments to 2.
      // We can invoke it through the component's internal state by calling ngOnInit pattern.
      // Since loadPage is private, we call it via bracket notation.
      await (component as any).loadPage();
      expect(component.photos()).toHaveLength(2);
      expect(mockApi.get).toHaveBeenCalledWith('/persons/42/photos', { page: 2, per_page: 48 });
    });

    it('should set allLoaded when all photos are fetched', async () => {
      const response = {
        photos: [
          { path: '/a.jpg', filename: 'a.jpg', aggregate: 8, aesthetic: 7, date_taken: '' },
          { path: '/b.jpg', filename: 'b.jpg', aggregate: 7, aesthetic: 6, date_taken: '' },
        ],
        total: 2,
        person: { id: 42, name: 'Alice', face_count: 10 },
      };
      mockApi.get.mockReturnValue(of(response));

      await component.ngOnInit();

      // allLoaded should be true, so subsequent loadPage calls are no-ops
      mockApi.get.mockClear();
      await (component as any).loadPage();
      expect(mockApi.get).not.toHaveBeenCalled();
    });

    it('should not call API when already loading', async () => {
      let resolveGet!: (value: any) => void;
      mockApi.get.mockReturnValue(new Promise((r) => (resolveGet = r)));

      // Start first load (will set loading=true)
      const loadPromise = (component as any).loadPage();

      // Try second load while first is pending
      mockApi.get.mockReturnValue(of(mockPhotosResponse));
      await (component as any).loadPage();

      // Only 1 call should have been made
      expect(mockApi.get).toHaveBeenCalledTimes(1);

      // Clean up by resolving
      resolveGet(of(mockPhotosResponse));
      await loadPromise.catch(() => {});
    });

    it('should handle API errors gracefully and set allLoaded', async () => {
      mockApi.get.mockReturnValue(throwError(() => new Error('Network error')));

      await component.ngOnInit();

      expect(component.loading()).toBe(false);
      // After error, allLoaded=true so no more calls
      mockApi.get.mockClear();
      mockApi.get.mockReturnValue(of(mockPhotosResponse));
      await (component as any).loadPage();
      expect(mockApi.get).not.toHaveBeenCalled();
    });
  });
});
